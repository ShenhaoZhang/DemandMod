from itertools import product
from functools import reduce
from string import ascii_lowercase,ascii_uppercase

import numpy as np 
import pandas as pd 

class Sale:
    def __init__(self,attrs_num=[2,2],theta_attrs_f=None,theta_attrs_pi=None,seed=1) -> None:
        self.attrs_num      = attrs_num
        self.theta_attrs_f  = theta_attrs_f
        self.theta_attrs_pi = theta_attrs_pi
        self.rng            = np.random.default_rng(seed)
        
        self.attrs_type     = None
        self.theta_goods_f  = None
        self.theta_goods_pi = None
        self.__init_attrs()
        self.__init_theta()
        
        self.data_origin     = None
        self.data_trans      = None
        self.data_trans_prob = None
        self.data_observed   = None
    
    def __init_attrs(self):
        self.attrs_type = {}
        for attr_index in range(len(self.attrs_num)):
            attr = list(ascii_uppercase)[attr_index]
            attr_levels = list(ascii_lowercase)[0:self.attrs_num[attr_index]]
            self.attrs_type[attr] = attr_levels
            
        self.attrs_type_flatten = []
        for attr,level in self.attrs_type.items():
            self.attrs_type_flatten.append(list(map(lambda x:attr+'_'+x,level)))
        self.goods_type = list(map(lambda attrs:reduce(lambda x,y:x+'*'+y,attrs),product(*self.attrs_type_flatten)))
        
    
    def __init_theta(self):
        if self.theta_attrs_f is None:
            self.theta_attrs_f = []
            for attr_index in range(len(self.attrs_num)):
                theta_attr_f = self.rng.uniform(0,1,size=self.attrs_num[attr_index])
                theta_attr_f = np.exp(theta_attr_f) / np.sum(np.exp(theta_attr_f))
                self.theta_attrs_f.append(theta_attr_f)
        else:
            if len(self.theta_attrs_f) != len(self.attrs_num):
                raise Exception('theta_attrs_f参数有误')
            for attr_index in range(len(self.attrs_num)):
                if len(self.theta_attrs_f[attr_index]) != self.attrs_num[attr_index]:
                    raise Exception('theta_attrs_f参数有误')
                if sum(self.theta_attrs_f[attr_index]) != 1:
                    raise Exception('theta_attrs_f参数有误')
            self.theta_attrs_f = list(map(lambda x:np.array(x),self.theta_attrs_f))
        
        if self.theta_attrs_pi is None:
            self.theta_attrs_pi = []
            for attr_index in range(len(self.attrs_num)):
                nrow = self.attrs_num[attr_index]
                theta_attr_pi = self.rng.uniform(0,1,size=nrow**2).reshape(nrow,-1)
                theta_attr_pi[np.diag_indices_from(theta_attr_pi)] = 1
                self.theta_attrs_pi.append(theta_attr_pi)
        else:
            if len(self.theta_attrs_pi) != len(self.attrs_num):
                raise Exception('theta_attrs_pi参数有误')
            #TODO 增加验证
            self.theta_attrs_pi = list(map(lambda x:np.array(x),self.theta_attrs_pi))
        
        self.theta_goods_f = np.array(np.meshgrid(*self.theta_attrs_f)).T.reshape(-1,len(self.attrs_num)).prod(axis=1)
        self.theta_goods_pi = reduce(np.kron,self.theta_attrs_pi)
        
    def sim(self,unsale_pct=0.25,size=100,n=1000):
        # 商品齐全时的销售数据
        origin_data = pd.DataFrame(data=self.rng.multinomial(n=n,pvals=self.theta_goods_f,size=size),
                                   columns=self.goods_type)
        
        # 实际未售和实际在售的商品索引
        unsale_goods_count = round(len(self.goods_type)*unsale_pct)
        sale_goods_index   = slice(0,-unsale_goods_count)
        unsale_goods_index = slice(-unsale_goods_count,10000)
        
        # 商品齐全时两部分商品的数据
        unsale_goods_data = origin_data.loc[:,self.goods_type[unsale_goods_index]]
        sale_goods_data = origin_data.loc[:,self.goods_type[sale_goods_index]]
        
        # 转移的销售额
        ## 转移矩阵
        trans_prob = self.get_trans_prob(self.theta_goods_pi[unsale_goods_index,sale_goods_index])
        trans_sale = np.dot(unsale_goods_data.to_numpy(),trans_prob)
        trans_data = pd.DataFrame(data=trans_sale,columns=self.goods_type[sale_goods_index]).round().astype('int')
        
        observed_goods_data = sale_goods_data + trans_data
        observed_goods_data = observed_goods_data.reindex(origin_data.columns,axis=1)
        
        self.data_origin     = origin_data
        self.data_trans      = trans_data
        self.data_trans_prob = pd.DataFrame(data=trans_prob,columns=self.goods_type[sale_goods_index],index=self.goods_type[unsale_goods_index])
        self.data_observed   = observed_goods_data
    
    @staticmethod
    def get_trans_prob(matrix):
        matrix_max = np.zeros_like(matrix)
        n_row,n_col = matrix_max.shape
        for row_index in range(n_row):
            col_max_index = np.argmax(matrix[row_index])
            for col_index in range(n_col):
                if col_index == col_max_index:
                    matrix_max[row_index,col_index] = matrix[row_index,col_index]
                else:
                    pass
        return matrix_max

if __name__ == '__main__':
    s = Sale(attrs_num=[2,2],
             theta_attrs_f=[[0.1,0.9],[0.3,0.7]],
             theta_attrs_pi=[[[1,0.3],[0.4,1]],[[1,0.5],[0.6,1]]])
    print(s.attrs_type)
    print(s.attrs_type_flatten)
    print(s.goods_type)
    print(s.theta_attrs_f)
    print(s.theta_attrs_pi[0])
    print(s.theta_attrs_pi[1])
    print(s.theta_goods_f)
    print(s.theta_goods_pi)
    print(s.sim())