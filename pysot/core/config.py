from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "phtrack_r50"

__C.CUDA = True

__C.TRAIN = CN()

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 2.0

__C.TRAIN.CEN_WEIGHT = 1.0

__C.TRAIN.CG_WEIGHT = 0.1

__C.TRAIN.PRINT_FREQ = 10

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

__C.DATASET = CN(new_allowed=True)

__C.DATASET.TEMPLATE = CN()

__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'GOT')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = ''
__C.DATASET.VID.ANNO = ''
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = -1

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = r'G:\0hsi_train_test_data\hsi_train\got10k\crop_hsi511'
__C.DATASET.GOT.ANNO = r'G:\0hsi_train_test_data\hsi_train\got10k\train.json'

__C.DATASET.GOT.FRAME_RANGE = 50
__C.DATASET.GOT.NUM_USE = 3000
__C.DATASET.VIDEOS_PER_EPOCH = 3000

__C.BS = CN()
__C.BS.SELECT = True
__C.BS.TYPE = ""

__C.FS = CN()
__C.FS.FUSION = True
__C.FS.TYPE = ""

__C.BACKBONE = CN()

__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

__C.BACKBONE.PRETRAINED = ''

__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

__C.BACKBONE.LAYERS_LR = 0.1

__C.BACKBONE.TRAIN_EPOCH = 30

__C.ADJUST = CN()

__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

__C.ADJUST.TYPE = "AdjustAllLayer"

__C.CAR = CN()

__C.CAR.TYPE = 'MultiCAR'

__C.CAR.KWARGS = CN(new_allowed=True)

__C.TRACK = CN()

__C.TRACK.TYPE = 'phtrackTracker'

__C.TRACK.PENALTY_K = 0.04

__C.TRACK.WINDOW_INFLUENCE = 0.44

__C.TRACK.LR = 0.4

__C.TRACK.EXEMPLAR_SIZE = 127

__C.TRACK.INSTANCE_SIZE = 255

__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8

__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44

__C.HP_SEARCH = CN()

__C.HP_SEARCH.OTB100 = [0.35, 0.45, 0.45, 0.508]
