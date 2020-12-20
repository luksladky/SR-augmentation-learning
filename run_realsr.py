print('Run experiment with RealSR dataset (153 training images).')

from srthesis.utils import *
select_gpu()

from fastai.utils.mem import *
from srthesis.data_realsr import *
from srthesis.training import *

gc.collect()
run_info = RunInfo(group_id = 'realsr', 
                   run_id = 'realsr_orig', 
                   dataset = 'realsr',
                   repeat_train= 10, 
                   pretraining=False, 
                   dihedral_augs=True,
                   loss_type='feature')
run_experiment(run_info, num_workers=16, start_epoch=0)