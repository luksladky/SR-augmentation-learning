print('Run experiment with RealSR dataset (153 training images).')

from srthesis.utils import *
select_gpu()

from fastai.utils.mem import *
from srthesis.training import *

gc.collect()
run_experiment('realsr_full', 'realsr', start_epoch=0, checkpoint_every=50000, loss_type='l1')