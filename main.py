from model import *
from config import *


cfg = base_model_config()


mm = Pricing('bosch', cfg)

mm.profit_model()

