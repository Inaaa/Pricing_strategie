import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def base_model_config():

    cfg = edict()

    #linear price menge function
    cfg.LINEAR_A = 3 #TODO
    cfg.LINEAR_B = 2 #TODO

    #Gutenberg function --price and menge
    cfg.GUTENBERG_A = 10 #todo
    cfg.GUTENBERG_C1 = 3 #TODO
    cfg.GUTENBERG_C2 = 1 #TODO



    return cfg
