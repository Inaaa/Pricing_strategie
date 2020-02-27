import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def two_side_config():

    cfg = edict()
    cfg.A = 0  # u = a +bn
    cfg.B = 5  # u = a +bn
    cfg.C = 5  # p = c -dn
    cfg.D = 0.5  # p = c -dn
    cfg.INFLECTION_POINT = 10
    cfg.PU = 5  # consumer charge profi to platform
    cfg.THETA = 5
    cfg.PD = 10 # producer charge profi to platform
    cfg.PHI = 5

    return cfg




