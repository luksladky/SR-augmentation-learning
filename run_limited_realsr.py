from srthesis.utils import *
select_gpu()

from fastai.utils.mem import *
from srthesis.training import *

train_len = 1
print(f'Run experiment with limited RealSR dataset ({train_len} training images, no augmentations) 2.')

gc.collect()
#run_experiment('realsr_limited_1', 'realsr', dihedral_augs=False, max_train_len=1, checkpoint_every=1000)
run_overfitting('realsr_limited_1_third',max_train_len=train_len, checkpoint_every=50000)