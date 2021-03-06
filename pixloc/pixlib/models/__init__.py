from ..utils.tools import get_class
from .base_model import BaseModel

# get_model(conf.model.name)(conf.model).to(device)
# name: two_view_refiner
# conf.model:model関係のconf
# __name__:Python のプログラムがどこから呼ばれて実行されているかを格納している変数
def get_model(name):
    return get_class(name, __name__, BaseModel)
