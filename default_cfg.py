from yacs.config import CfgNode as CN 

cfg = CN()
# ---------------------------------------------------------------------------- #
# Global options
# ---------------------------------------------------------------------------- #
cfg.OUTPUT_DIR = 'work_dirs'
cfg.TOTAL_EPOCHS = 1000
cfg.SAVE_CKPT_AFTER_EPOCH = 0
cfg.SAVE_CKPT_EVERY_EPOCH = 100 
cfg.MODEL = CN()
cfg.MODEL.TYPE_NAME = 'TextRecModel'
cfg.MODEL.TOTAL_DEPTHS = 18  # for searching depth, only valid for 'TextRecModel_SuperDepth' model 
cfg.MODEL.CH_SCALE_RATIO = 1.0
cfg.MODEL.IN_CHANNELS = 1  # 1 for gray, 3 for rgb
# ---------------------------------------------------------------------------- #
# Backbone (CNN)
# ---------------------------------------------------------------------------- #
cfg.MODEL.CNN = CN()
cfg.MODEL.CNN.TYPE_NAME = ''
cfg.MODEL.CNN.STRIDES     = [[2, 2], [2, 2], [2, 1], [2, 1], [2, 1]]
cfg.MODEL.CNN.LAYERS      = [3, 3, 4, 3, 3]
cfg.MODEL.CNN.BLOCKS      = ['BasicBlock', 'SlimBasicBlock', 'BasicBlock', 'BasicBlock', 'BasicBlock']
cfg.MODEL.CNN.NOLINEARS_1   = [0, 0, 0, 0, 0]
cfg.MODEL.CNN.NOLINEARS_2   = [0, 0, 0, 0, 0]
cfg.MODEL.CNN.SES         = [0, 0, 0, 0, 0]

# ---------------------------------------------------------------------------- #
# RNN
# ---------------------------------------------------------------------------- #
cfg.MODEL.RNN = CN()
cfg.MODEL.RNN.TYPE_NAME = 'Transformer'
cfg.MODEL.RNN.TRANSFORMER = CN()
cfg.MODEL.RNN.TRANSFORMER.DROPOUT     = 0.1
cfg.MODEL.RNN.TRANSFORMER.NUM_LAYERS    = 1
cfg.MODEL.RNN.TRANSFORMER.NUM_HEADERS   = 8
cfg.MODEL.RNN.TRANSFORMER.ATTN_BLOCK_TYPES = ['ATTN'] 
cfg.MODEL.RNN.TRANSFORMER.ATTN_BLOCK_TYPE = 'ATTN'
cfg.MODEL.RNN.TRANSFORMER.FFN_BLOCK_TYPES   = ['MLP_4']
cfg.MODEL.RNN.TRANSFORMER.FFN_BLOCK_TYPE   = 'MLP_4'

# ---------------------------------------------------------------------------- #
# Prediction HEAD 
# ---------------------------------------------------------------------------- #
cfg.MODEL.HEAD = CN()
cfg.MODEL.HEAD.TYPE_NAME   = 'MLPHead'
cfg.MODEL.HEAD.IN_CHANNELS = 512 
cfg.MODEL.HEAD.NUM_CLASSES = 80 

# ---------------------------------------------------------------------------- #
# Optimizer                                                                                   
# ---------------------------------------------------------------------------- #
cfg.OPTIM = CN()
cfg.OPTIM.CLASS_NAME = 'Adadelta'
cfg.OPTIM.INIT_LR = 1.0 
cfg.OPTIM.WEIGHT_DECAY = 1e-5
cfg.OPTIM.N_WARMUP_EPOCHS = -1 

# ---------------------------------------------------------------------------- #
# Dataset                                                                                   
# ---------------------------------------------------------------------------- #
cfg.DATASETS = CN()
cfg.DATASETS.CASE_SENSITIVE = True 
cfg.DATASETS.DICT_FILE = 'data/IAM/dic_79.txt'
cfg.DATASETS.ROOT_DIR = '/data/DB_HW/formatted/IAM_lines'
cfg.DATASETS.TRAIN = CN()
cfg.DATASETS.TRAIN.CLASS_NAME = 'IAMDataset'
cfg.DATASETS.TRAIN.IMG_LIST_FILE = 'data/IAM/train_list.txt'
cfg.DATASETS.TRAIN.INPUT_SHAPE = [1, 64, 1200]
cfg.DATASETS.TRAIN.BATCH_SIZE = 32
cfg.DATASETS.TRAIN.INIT_P_AUG = 0.5 
cfg.DATASETS.TRAIN.WORD_AUG = False 
cfg.DATASETS.TRAIN.IAM_PATH = 'data'

cfg.DATASETS.VAL = CN()
cfg.DATASETS.VAL.CLASS_NAME = 'IAMDataset'
cfg.DATASETS.VAL.IMG_LIST_FILE = 'data/IAM/val_list.txt'
cfg.DATASETS.VAL.INPUT_SHAPE = [1, 64, 1200]
cfg.DATASETS.VAL.BATCH_SIZE = 32 

cfg.DATASETS.TEST = CN()
cfg.DATASETS.TEST.CLASS_NAME = 'IAMDataset'
cfg.DATASETS.TEST.IMG_LIST_FILE = 'data/IAM/test_list.txt'
cfg.DATASETS.TEST.INPUT_SHAPE = [1, 64, 1200]
cfg.DATASETS.TEST.BATCH_SIZE = 32 
