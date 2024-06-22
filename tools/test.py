from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import cv2
import torch
import numpy as np
from pysot.core.config import cfg
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
from toolkit.datasets import DatasetFactory
import math
import logging
from pysot.utils.log_helper import init_log
import matplotlib.pyplot as plt
import matplotlib
import sys

sys.path.append('../')
logger = logging.getLogger('global')
init_log('global', logging.INFO)
imageFilter = '*.png*'
parser = argparse.ArgumentParser(description='siamcar tracking')
parser.add_argument('--dataset', type=str, default='OTB100', help='datasets')
parser.add_argument('--video', default='', type=str, help='test one special video')
parser.add_argument('--snapshot', type=str, default='snapshot/a.pth', help='snapshot of models to eval')
parser.add_argument('--config', type=str, default='./config.yaml', help='config file')
parser.add_argument('--vis', action='store_true', default=False, help='whether visualzie result')
args = parser.parse_args()
torch.set_num_threads(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def main():
    cfg.merge_from_file(args.config)
    params = getattr(cfg.HP_SEARCH, args.dataset)
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2], 'context_amount': params[3]}
    dataset_root = r'G:\0hsi_train_test_data\hsi_test\OTB100'  #
    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot).cuda().eval()
    tracker = SiamCARTracker(model, cfg.TRACK)
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root, load_img=False)
    model_name = args.snapshot.split('.')[-2].split('/')[-1] + '_lr' + str(hp['lr']) + '_pk' + str(
        hp['penalty_k']) + '_wlr' + str(hp['window_lr']) + '_ca' + str(hp['context_amount'])
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            if video.name != args.video:
                continue
        pred_bboxes = []
        track_times = []
        w_std = []
        for idx, (img, gt_bbox) in enumerate(video):
            img = img / img.max() * 255
            img = img.astype('uint8')
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
                tracker.init(img, gt_bbox_, hp)
            else:
                outputs = tracker.track(img, hp)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                w_std.append(outputs['w_std'].item())
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fps_path = os.path.join('results', args.dataset, 'fps')
        if not os.path.isdir(fps_path):
            os.makedirs(fps_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        result_fps_path = os.path.join(fps_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        with open(result_fps_path, 'w') as f:
            for x in track_times:
                f.write(str(x) + '\n')
        fps = len(track_times) / sum(track_times)
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx + 1, video.name, sum(track_times),
                                                                              fps))


if __name__ == '__main__':
    main()
