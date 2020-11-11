train_len = 1
print(f'Run experiment with limited RealSR dataset ({train_len} training images, no augmentations) L1 loss only')

from srthesis.utils import *
select_gpu()

from fastai.utils.mem import *
from srthesis.training import *

gc.collect()

run_overfitting('realsr_limited_1_l1_loss',max_train_len=train_len, start_epoch=0, checkpoint_every=50000, random_seed=68, loss_type='l1')

run_overfitting('realsr_limited_1',max_train_len=train_len, start_epoch=0, checkpoint_every=50000, random_seed=68, loss_type='feature')
