from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import logging
import os
import glob
import time
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, reduce_gradients, average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg
import sys
import warnings

CUDA_VISIBLE_DEVICES = [1, 2]  # 2 cards

sys.path.append('../')
warnings.filterwarnings("ignore")
logger = logging.getLogger('global')

parser = argparse.ArgumentParser(description='phtrack training')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')

parser.add_argument('--cfg', type=str, default='../experiments/phtrack/config.yaml', help='configuration')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("build train dataset")
    train_dataset = TrkDataset()
    logger.info("build dataset done")
    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True, sampler=train_sampler)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for param in model.neck.parameters():
        param.requires_grad = False
    for m in model.neck.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for param in model.car_head.parameters():
        param.requires_grad = False
    for m in model.car_head.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for param in model.down.parameters():
        param.requires_grad = False
    for m in model.down.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    trainable_params = []
    trainable_params += [{'params': model.channelsplit.parameters(), 'lr': cfg.TRAIN.BASE_LR}]
    trainable_params += [{'params': model.fs.parameters(), 'lr': cfg.TRAIN.BASE_LR}]

    params_train = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_train.append(name)
    logger.info('Trainable(可训练层) layers:{}'.format(params_train))

    optimizer = torch.optim.SGD(trainable_params, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)

    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, car_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            car_norm += _norm ** 2
        tb_writer.add_scalar('grad_all/' + k.replace('.', '/'), _norm, tb_index)
        tb_writer.add_scalar('weight_all/' + k.replace('.', '/'), w_norm, tb_index)
        tb_writer.add_scalar('w-g/' + k.replace('.', '/'), w_norm / (1e-20 + _norm), tb_index)
    tot_norm = feature_norm + car_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    car_norm = car_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/car', car_norm, tb_index)


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()
    average_meter = AverageMeter()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)
    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()

    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch
            if get_rank() == 0:
                torch.save(
                    {'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()},
                    cfg.TRAIN.SNAPSHOT_DIR + '/checkpoint_e%d.pth' % epoch)

            if epoch == cfg.TRAIN.EPOCH:
                return
            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))
            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch + 1))

        tb_idx = idx

        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {}, lr {}'.format(epoch + 1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx + 1), pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        outputs = model(data)

        loss = outputs['total_loss'].mean()

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()

            loss.backward()

            reduce_gradients(model)
            if rank == 0 and cfg.TRAIN.LOG_GRADS:
                log_grads(model.module, tb_writer, tb_idx)
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)

            optimizer.step()

        batch_time = time.time() - end
        batch_info = {'batch_time': average_reduce(batch_time), 'data_time': average_reduce(data_time)}
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.mean().data.item())
        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}], [{}/{}(每epoch的loader长度)] lr(当前学习率): {:.6f}\n".format(epoch + 1, (
                        idx + 1) % num_per_epoch, num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += "\t{:s}\t".format(getattr(average_meter, k))
                    else:
                        info += "{:s}\n".format(getattr(average_meter, k))
                logger.info(info)
                print_speed(idx + 1 + start_epoch * num_per_epoch, average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()


def main():
    rank, world_size = dist_init()
    print('\n')
    logger.info("init done")

    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'), logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    model = ModelBuilder()
    model.train()
    dist_model = nn.DataParallel(model, device_ids=CUDA_VISIBLE_DEVICES).cuda()

    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    train_loader = build_data_loader()

    optimizer, lr_scheduler = build_opt_lr(dist_model.module, cfg.TRAIN.START_EPOCH)

    cur_path1 = os.path.dirname(os.path.realpath(__file__))

    model_general_path = os.path.join(cur_path1, '../', 'pretrained_models', cfg.TRAIN.PRETRAINED)
    load_pretrain(model, model_general_path)

    if cfg.TRAIN.RESUME:

        checkpoint_list = sorted(glob.glob('{}/{}/{}_e*.pth'.format(cur_path1, 'snapshot', 'checkpoint')))
        if len(checkpoint_list) != 0:
            checkpoint_path = checkpoint_list[-1]

            reuse_model_path = os.path.join(cur_path1, 'snapshot', checkpoint_path)
            logger.info("resume from {}".format(checkpoint_path))

            assert os.path.isfile(reuse_model_path), '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
            model, optimizer, cfg.TRAIN.START_EPOCH = restore_from(model, optimizer, reuse_model_path)

    dist_model = nn.DataParallel(model, device_ids=CUDA_VISIBLE_DEVICES).cuda()

    device = torch.device("cuda:1")
    dist_model.to(device)

    print('----------------------------------------')
    logger.info(lr_scheduler)
    logger.info("model prepare done")

    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
