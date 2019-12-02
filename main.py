from model import *
from config import *


cfg = base_model_config()
a =cfg.LINEAR_A

mm = Pricing('bosch', cfg)

mm.profit_model()

