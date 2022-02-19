from functools import reduce
from itertools import product
from string import ascii_lowercase,ascii_uppercase

import numpy as np
import pandas as pd

class SimSale:
    """
    通过需求模型模拟数据
    """
    
    def __init__(self,level_size=[3,3],seed=123) -> None:
        """
        初始化

        Parameters
        ----------
        level_size : list, optional
            各个属性下相应水平的数量, by default [3,3]
        """
        self.rng = np.random.default_rng(seed=seed)
        self.level_size = level_size
        self.attr_size = len(self.level_size)
        self.attr_name = list(ascii_uppercase)[0:self.attr_size]
        self.level_name = list(map(lambda x:list(ascii_lowercase)[0:x],level_size))
        
        self.attr_f_list = None
        self.attr_pi_list = None
        self.goods_f = None
        self.goods_pi = None
        self.generate_attr_param()
        self.generate_goods_param()
        
    
    def generate_attr_param(self):
        """
        属性选择及转移参数
        """
        self.attr_f_list = []
        self.attr_pi_list = []
        for i in range(self.attr_size):
            # 直接选择概率
            attr_f = self.rng.uniform(low=0,high=1,size=self.level_size[i])
            attr_f = attr_f / np.sum(attr_f)
            attr_f = pd.Series(data=attr_f,
                               name=self.attr_name[i],
                               index=self.level_name[i])
            self.attr_f_list.append(attr_f)
            
            # 间接转移概率
            attr_pi = self.rng.uniform(low=0,
                                       high=0.99,
                                       size=[self.level_size[i],self.level_size[i]])
            attr_pi[np.diag_indices_from(attr_pi)] = 1
            attr_pi = pd.DataFrame(data=attr_pi,
                                   index=self.level_name[i],
                                   columns=self.level_name[i])
            self.attr_pi_list.append(attr_pi)
    
    def generate_goods_param(self):
        """
        通过属性计算商品的选择及转移参数
        """
        # 直接选择概率
        goods_f = map(lambda pair:reduce(lambda x,y:x*y,pair), product(*self.attr_f_list))
        goods_type = map(lambda sr:[sr.name+'_'+v for v in sr.index],self.attr_f_list)
        goods_f_name = map(lambda pair:reduce(lambda x,y:x+'*'+y,pair), product(*goods_type))
        self.goods_f = pd.Series(data=list(goods_f),index=list(goods_f_name))
        
        # 间接转移概率
        self.goods_pi = pd.DataFrame(
            data=reduce(np.kron,map(lambda sr:sr.to_numpy(),self.attr_pi_list)),
            index=self.goods_f.index,
            columns=self.goods_f.index
        )
    
    def generate_single_sale(self,n=100,un_ava_frac=0.1,un_ava_mix=True):
        """
        计算单个样本

        Parameters
        ----------
        n : int, optional
            多项式分布中的n,代表所有商品的总销量, by default 100
        un_ava_frac : float, optional
            缺货或未分档的商品比例, by default 0.1

        Returns
        -------
        DataFrame
            单个样本
        """
        # 所有销量数据
        all_sale = pd.Series(data=self.rng.multinomial(n=n, pvals=self.goods_f.to_numpy()),
                             index=self.goods_f.index)
        
        # 由于缺货或未分档而未直接产生的销量
        if un_ava_mix :
            un_ava_sale = all_sale.sample(frac=un_ava_frac,replace=False)
        else:
            un_ava_sale = all_sale.tail(int(len(self.goods_f)*un_ava_frac))
            
        # 不考虑转移的情况下的销量
        ava_sale = all_sale[~all_sale.index.isin(un_ava_sale.index)]
        
        # 转移矩阵
        pi = self.goods_pi.loc[un_ava_sale.index,ava_sale.index]
        pi_matrix = pi.to_numpy()
        pi_matrix_max = np.zeros_like(pi_matrix)
        row_indices = np.arange(pi_matrix.shape[0])
        col_indices = np.argmax(pi_matrix,axis=1)
        pi_matrix_max[row_indices,col_indices] = pi_matrix[row_indices,col_indices]
        
        # 转移的销量
        pi_sale = np.dot(un_ava_sale.to_numpy(),pi_matrix_max).astype(int) + ava_sale.to_numpy()
        
        # 真实销量
        real_sale = pd.DataFrame(data=[pi_sale],columns=ava_sale.index)
        return real_sale
    
    def generate_sale(self,size=100,lam=100,un_ava_frac=0.1,un_ava_mix=True):
        """
        多次观测的样本

        Parameters
        ----------
        size : int, optional
            样本量, by default 100
        lam : int, optional
            每次总销量的泊松分布的均值, by default 100

        Returns
        -------
        DataFrame
            销售数据
        """
        obs_list = []
        for i in range(size):
            n = self.rng.poisson(lam=lam)
            obs = self.generate_single_sale(n = n,
                                            un_ava_frac = un_ava_frac,
                                            un_ava_mix = un_ava_mix)
            obs_list.append(obs)
        sale = pd.concat(obs_list)
        sale = sale.reindex(self.goods_f.index,axis=1)
        return sale