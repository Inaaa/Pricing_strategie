import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def base_model_config():

    cfg = edict()

    #choose optimize funktion
    cfg.OPT_FUNTION = 'gutenberg'  # linear , oligopol_linear

    # choose the reaction hypothese to the competition company
    cfg.REACTION_HYPOTHESE = 'Cournot'  # chamberlin ,  stackelberg

    # choose the quantity-related price difference
    cfg.TARIF = 'zweiteilig'  # block, durchgerechnet,
    # define the menge qb
    cfg.TARIF_QB = 20
    cfg.TARIF_DISCOUNT = 15

    # fixed grundgebühr
    cfg.FIXED_GRUNDKOST = 10   # TODO

    #linear price menge function
    cfg.LINEAR_A = 3 #TODO
    cfg.LINEAR_B = 2 #TODO
    cfg.LINEAR_C = 1 # TODO

    #linear price menge function of competition company
    cfg.COM_AVERAGE_PRICE = 20  # TODO
    cfg.COM_LINEAR_ALPHA = 1   # TODO
    cfg.COM_LINEAR_BETA = 2   # TODO


    #Gutenberg function --price and menge
    cfg.GUTENBERG_A = 10 #todo
    cfg.GUTENBERG_C1 = 3 #TODO
    cfg.GUTENBERG_C2 = 1 #TODO




    #





    return cfg
