from srthesis.utils import *
#select_gpu()

from fastai.utils.mem import *
from srthesis.training import *

gc.collect()
#run_experiment('realsr_limited_1', 'realsr', dihedral_augs=False, max_train_len=1, checkpoint_every=1000)
#run_overfitting('realsr_limited_1_third',max_train_len=train_len, checkpoint_every=50000)

fnames_plants = ['datasets/RealSR/Canon/Train/2/Canon_021_LR2.png', 
          'datasets/RealSR/Canon/Train/2/Canon_054_LR2.png',
 'datasets/RealSR/Nikon/Train/2/Nikon_080_LR2.png',
 'datasets/RealSR/Nikon/Train/2/Nikon_024_LR2.png',
 'datasets/RealSR/Canon/Train/2/Canon_047_LR2.png']

fnames_building = ['datasets/RealSR/Canon/Train/2/Canon_022_LR2.png',
'datasets/RealSR/Canon/Train/2/Canon_029_LR2.png',
'datasets/RealSR/Canon/Train/2/Canon_030_LR2.png',
'datasets/RealSR/Canon/Train/2/Canon_044_LR2.png',
'datasets/RealSR/Nikon/Train/2/Nikon_012_LR2.png']

train_len = 0

run_info = RunInfo(group_id = 'limited', 
                   run_id = 'realsr_limited_5_buildings', 
                   dataset = 'realsr',
                   max_train_len=train_len,
                   repeat_train=1000, 
                   pretraining=False, 
                   dihedral_augs=False,
                   train_include=fnames_building,
                   loss_type='feature')

print(f'Super-Resolution: Limited RealSR dataset ({train_len} training images, no augmentations)')
run_experiment(run_info, num_workers=16, start_epoch=0)