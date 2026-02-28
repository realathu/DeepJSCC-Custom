# -*- coding: utf-8 -*-
"""
Created on Tue Dec  17:00:00 2023

@author: chun
"""
import os
import multiprocessing
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import DeepJSCC, ratio2filtersize
from torch.nn.parallel import DataParallel
from utils import image_normalization, set_seed, save_model, view_model_param
from fractions import Fraction
from dataset import Vanilla
import numpy as np
import time
from tensorboardX import SummaryWriter
import glob

# AMP — use BF16 on A100 (native format, no overflow risk vs FP16)
_AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# TF32 — A100 tensor-core native format; enabled for both matmul and convolutions.
# Gives 2-3x matmul throughput vs FP32 with negligible precision loss.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32      = True
torch.set_float32_matmul_precision('high')


def train_epoch(model, optimizer, param, data_loader, scaler):
    """One training epoch with BF16 Automatic Mixed Precision."""
    model.train()
    epoch_loss = 0
    device = param['device']
    denorm = image_normalization('denormalization')  # build closure once per epoch

    for iter, (images, _) in enumerate(data_loader):
        images = images.cuda() if param['parallel'] and torch.cuda.device_count(
        ) > 1 else images.to(device)
        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

        with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
            outputs = model.forward(images)
            outputs_d = denorm(outputs)
            images_d  = denorm(images)
            loss = (model.loss(images_d, outputs_d) if not param['parallel']
                    else model.module.loss(images_d, outputs_d))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    return epoch_loss, optimizer


def evaluate_epoch(model, param, data_loader):
    """Validation loop — runs in BF16 autocast for consistency."""
    model.eval()
    epoch_loss = 0
    device = param['device']
    denorm = image_normalization('denormalization')  # build closure once per call

    with torch.no_grad():
        for iter, (images, _) in enumerate(data_loader):
            images = images.cuda() if param['parallel'] and torch.cuda.device_count(
            ) > 1 else images.to(device)
            with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                outputs = model.forward(images)
                outputs_d = denorm(outputs)
                images_d  = denorm(images)
                loss = (model.loss(images_d, outputs_d) if not param['parallel']
                        else model.module.loss(images_d, outputs_d))
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

    return epoch_loss


def config_parser_pipeline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'imagenet'], help='dataset')
    parser.add_argument('--out', default='./out', type=str, help='out_path')
    parser.add_argument('--disable_tqdm', default=False, type=bool, help='disable_tqdm')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--parallel', default=False, type=bool, help='parallel')
    parser.add_argument('--snr_list', default=['19', '13',
                        '7', '4', '1'], nargs='+', help='snr_list')
    parser.add_argument('--ratio_list', default=['1/6', '1/12'], nargs='+', help='ratio_list')
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel')

    return parser.parse_args()


def main_pipeline():
    args = config_parser_pipeline()

    print("Training Start")
    dataset_name = args.dataset
    out_dir = args.out
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    params = {}
    params['disable_tqdm'] = args.disable_tqdm
    params['dataset'] = dataset_name
    params['out_dir'] = out_dir
    params['device'] = args.device
    params['snr_list'] = args.snr_list
    params['ratio_list'] = args.ratio_list
    params['channel'] = args.channel
    # Determine optimal worker count: cap at 8, don't exceed CPU count
    _nw = min(8, multiprocessing.cpu_count())

    if dataset_name == 'cifar10':
        params['batch_size'] = 512        # A100 has ample VRAM; larger batch = faster epoch
        params['num_workers'] = _nw
        params['epochs'] = 1000
        params['init_lr'] = 1e-3
        params['weight_decay'] = 5e-4
        params['parallel'] = False
        params['if_scheduler'] = True
        params['seed'] = 42
        params['ReduceLROnPlateau'] = False
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 12
        params['min_lr'] = 1e-5
        params['es_patience'] = 50        # early stopping: max epochs without improvement
    elif dataset_name == 'imagenet':
        params['batch_size'] = 128         # up from 32; A100 handles this easily
        params['num_workers'] = _nw
        params['epochs'] = 300
        params['init_lr'] = 1e-4
        params['weight_decay'] = 5e-4
        params['parallel'] = True
        params['if_scheduler'] = True
        params['seed'] = 42
        params['ReduceLROnPlateau'] = True
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 12
        params['min_lr'] = 1e-5
        params['es_patience'] = 30        # early stopping: max epochs without improvement
    else:
        raise Exception('Unknown dataset')

    set_seed(params['seed'])

    for ratio in params['ratio_list']:
        for snr in params['snr_list']:
            params['ratio'] = ratio
            params['snr'] = snr

            train_pipeline(params)


# add train_pipeline to with only dataset_name args
def train_pipeline(params):

    dataset_name = params['dataset']
    # DataLoader kwargs shared by all loaders — optimised for A100
    _dl_kwargs = dict(
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        pin_memory=True,           # zero-copy host→device transfer
        persistent_workers=True,   # keep worker processes alive between epochs
        prefetch_factor=4,         # queue 4 batches per worker ahead of time
    )

    # load data
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        train_dataset = datasets.CIFAR10(root='../dataset/', train=True,
                                         download=True, transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True, **_dl_kwargs)

        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=False, **_dl_kwargs)  # no need to shuffle val

    elif dataset_name == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128
        print("loading data of imagenet")
        train_dataset = datasets.ImageFolder(root='../dataset/ImageNet/train', transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True, **_dl_kwargs)

        test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=False, **_dl_kwargs)  # no need to shuffle val
    else:
        raise Exception('Unknown dataset')

    # create model
    image_fisrt = train_dataset.__getitem__(0)[0]
    c = ratio2filtersize(image_fisrt, params['ratio'])
    print("The snr is {}, the inner channel is {}, the ratio is {:.2f}".format(
        params['snr'], c, params['ratio']))
    model = DeepJSCC(c=c, channel_type=params['channel'], snr=params['snr'])

    # init exp dir
    out_dir = params['out_dir']
    phaser = dataset_name.upper() + '_' + str(c) + '_' + str(params['snr']) + '_' + \
        "{:.2f}".format(params['ratio']) + '_' + str(params['channel']) + \
        '_' + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_log_dir = out_dir + '/' + 'logs/' + phaser
    root_ckpt_dir = out_dir + '/' + 'checkpoints/' + phaser
    root_config_dir = out_dir + '/' + 'configs/' + phaser
    writer = SummaryWriter(log_dir=root_log_dir)

    # model init
    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')
    if params['parallel'] and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()
    else:
        model = model.to(device)

    # torch.compile — fuses ops into CUDA kernels; ~20-40% throughput gain on A100
    # 'reduce-overhead' minimises Python overhead for the small per-batch kernel launches
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
        print("torch.compile enabled (reduce-overhead mode)")

    # GradScaler for AMP (identity-scaler for BF16, but kept for unified code path)
    scaler = torch.cuda.amp.GradScaler(enabled=(_AMP_DTYPE == torch.float16))

    # opt — AdamW correctly decouples weight decay from adaptive update (Adam doesn't)
    optimizer = optim.AdamW(
        model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])

    # Cosine Annealing — smooth LR decay over the full run; outperforms StepLR for JSCC
    if params['if_scheduler'] and not params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params['epochs'], eta_min=params['min_lr'])
    elif params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=params['lr_reduce_factor'],
                                                         patience=params['lr_schedule_patience'],
                                                         verbose=False)
    else:
        print("No scheduler")
        scheduler = None

    writer.add_text('config', str(params))
    t0 = time.time()
    epoch_train_losses, epoch_val_losses = [], []
    best_val_loss  = float('inf')  # tracks best val loss for checkpointing
    _es_best       = float('inf')  # separate tracker for early stopping
    _es_counter    = 0             # epochs without improvement
    per_epoch_time = []

    # train
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs']), disable=params['disable_tqdm']) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, optimizer = train_epoch(
                    model, optimizer, params, train_loader, scaler)

                epoch_val_loss = evaluate_epoch(model, params, test_loader)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss)

                per_epoch_time.append(time.time() - start)

                # Checkpoint — save best model + a periodic safety copy every 10 epochs.
                # Avoids the original behaviour of writing 1000 files to disk.
                os.makedirs(root_ckpt_dir, exist_ok=True)
                _state = (model.module.state_dict()
                          if isinstance(model, (DataParallel, torch.nn.parallel.DistributedDataParallel))
                          else model.state_dict())
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    torch.save(_state, root_ckpt_dir + '/best.pkl')
                if epoch % 10 == 0:
                    torch.save(_state, root_ckpt_dir + f'/epoch_{epoch}.pkl')

                if params['ReduceLROnPlateau'] and scheduler is not None:
                    scheduler.step(epoch_val_loss)
                elif params['if_scheduler'] and not params['ReduceLROnPlateau']:
                    scheduler.step()

                # Early stopping — count epochs without val loss improvement
                if epoch_val_loss < _es_best - 1e-6:
                    _es_best, _es_counter = epoch_val_loss, 0
                else:
                    _es_counter += 1
                    if _es_counter >= params.get('es_patience', 999):
                        print(f'\nEarly stopping triggered at epoch {epoch} '
                              f'(no improvement for {_es_counter} epochs)')
                        break

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(
                        params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    test_loss = evaluate_epoch(model, params, test_loader)
    train_loss = evaluate_epoch(model, params, train_loader)
    print("Test Accuracy: {:.4f}".format(test_loss))
    print("Train Accuracy: {:.4f}".format(train_loss))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    """
        Write the results in out_dir/results folder
    """

    writer.add_text(tag='result', text_string="""Dataset: {}\nparams={}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST Loss: {:.4f}\nTRAIN Loss: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""
                    .format(dataset_name, params, view_model_param(model), np.mean(np.array(train_loss)),
                            np.mean(np.array(test_loss)), epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))
    writer.close()
    if not os.path.exists(os.path.dirname(root_config_dir)):
        os.makedirs(os.path.dirname(root_config_dir))
    with open(root_config_dir + '.yaml', 'w') as f:
        dict_yaml = {'dataset_name': dataset_name, 'params': params,
                     'inner_channel': c, 'total_parameters': view_model_param(model)}
        import yaml
        yaml.dump(dict_yaml, f)

    del model, optimizer, scheduler, train_loader, test_loader
    del writer


def train(args, ratio: float, snr: float):  # deprecated

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # load data
    if args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        train_dataset = datasets.CIFAR10(root='../dataset/', train=True,
                                         download=True, transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128
        print("loading data of imagenet")
        train_dataset = datasets.ImageFolder(root='./dataset/ImageNet/train', transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        test_dataset = Vanilla(root='./dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise Exception('Unknown dataset')

    print(args)
    image_fisrt = train_dataset.__getitem__(0)[0]
    c = ratio2filtersize(image_fisrt, ratio)
    print("the inner channel is {}".format(c))
    model = DeepJSCC(c=c, channel_type=args.channel, snr=snr)

    if args.parallel and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()
        criterion = nn.MSELoss(reduction='mean').cuda()
    else:
        model = model.to(device)
        criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.if_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    epoch_loop = tqdm(range(args.epochs), total=args.epochs, leave=True, disable=args.disable_tqdm)
    for epoch in epoch_loop:
        run_loss = 0.0
        for images, _ in tqdm((train_loader), leave=False, disable=args.disable_tqdm):
            optimizer.zero_grad()
            images = images.cuda() if args.parallel and torch.cuda.device_count() > 1 else images.to(device)
            outputs = model(images)
            outputs = image_normalization('denormalization')(outputs)
            images = image_normalization('denormalization')(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        if args.if_scheduler:  # the scheduler is wrong before
            scheduler.step()
        with torch.no_grad():
            model.eval()
            test_mse = 0.0
            for images, _ in tqdm((test_loader), leave=False, disable=args.disable_tqdm):
                images = images.cuda() if args.parallel and torch.cuda.device_count() > 1 else images.to(device)
                outputs = model(images)
                images = image_normalization('denormalization')(images)
                outputs = image_normalization('denormalization')(outputs)
                loss = criterion(outputs, images)
                test_mse += loss.item()
            model.train()
        # epoch_loop.set_postfix(loss=run_loss/len(train_loader), test_mse=test_mse/len(test_loader))
        print("epoch: {}, loss: {:.4f}, test_mse: {:.4f}, lr:{}".format(
            epoch, run_loss / len(train_loader), test_mse / len(test_loader), optimizer.param_groups[0]['lr']))
    save_model(model, args.saved, args.saved + '/{}_{}_{:.2f}_{:.2f}_{}_{}.pth'
               .format(args.dataset, args.epochs, ratio, snr, args.batch_size, c))


def config_parser():  # deprecated
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2048, type=int, help='Random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=256, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel type')
    parser.add_argument('--saved', default='./saved', type=str, help='saved_path')
    parser.add_argument('--snr_list', default=['19', '13',
                        '7', '4', '1'], nargs='+', help='snr_list')
    parser.add_argument('--ratio_list', default=['1/3',
                        '1/6', '1/12'], nargs='+', help='ratio_list')
    parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'imagenet'], help='dataset')
    parser.add_argument('--parallel', default=False, type=bool, help='parallel')
    parser.add_argument('--if_scheduler', default=False, type=bool, help='if_scheduler')
    parser.add_argument('--step_size', default=640, type=int, help='scheduler')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma')
    parser.add_argument('--disable_tqdm', default=True, type=bool, help='disable_tqdm')
    return parser.parse_args()


def main():  # deprecated
    args = config_parser()
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    set_seed(args.seed)
    print("Training Start")
    for ratio in args.ratio_list:
        for snr in args.snr_list:
            train(args, ratio, snr)


if __name__ == '__main__':
    main_pipeline()
    # main()
