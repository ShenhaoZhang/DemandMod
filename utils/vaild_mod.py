from string import ascii_lowercase
import sys 
sys.path.append('..')

from mod.Demand import Demand
from mod.SimSale import SimSale

def vaild_mod(times = 30,level_size = [2,2],lam=1000,un_ava_frac=0.5,un_ava_mix=True):
    for i in range(times):
        sim_sale = SimSale(level_size=level_size,seed=i)
        data = sim_sale.generate_sale(lam=lam,un_ava_frac=un_ava_frac,un_ava_mix=un_ava_mix)
        
        mod = Demand(data=data,
                     goods_attr=map(lambda x:ascii_lowercase[0:x],level_size))
        