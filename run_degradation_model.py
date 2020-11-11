print('Train degradation model using experiment with RealSR dataset only.')

from srthesis.utils import *
select_gpu()

from fastai.utils.mem import *
from srthesis.training import *

gc.collect()
train_degradation_model('degradation_model')