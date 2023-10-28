from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from easydict import EasyDict

class Configuration:
    dataset = EasyDict({
        'data_path': '../dataset/xfund_funsd',
        'params': 'params_best/params.json',
        'features_path': 'features',
        'image_path': '../dataset/xfund_funsd',
        'scaler_path': 'weights',
        'model_path': 'weights',
        'debug_dir': 'debugs',
        'lang': 'pt'
        
    })
