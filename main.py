#from two_side import *
from config import *
from model import Pricing

cfg = base_model_config()

mm = Pricing('bosch', cfg)
mm.profit_model()

