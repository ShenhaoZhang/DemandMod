from itertools import product,accumulate,chain
from functools import reduce

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution,basinhopping,dual_annealing
import matplotlib.pyplot as plt
from numba import jit

class Demand:
    def __init__(self,data,goods_attr) -> None:
        self.data       = data
        self.goods_attr = goods_attr # {'price': ['l', 'm', 's'], 'size': ['l', 'm', 's']}
        
        self.attr_name         = None
        self.attr_type         = None
        self.attr_type_flatten = None
        self.goods_type        = None
        self.attr_trans        = None
        
        self._init_attr()
        self._init_attr_trans()
        
        self.theta_hat = None
        
    #TODO 增加data的初始化方法
    #TODO 增加通过先验确定转移参数
    #TODO 通过似然函数度量参数的不确定性
    #TODO 使用Numba加速
    
    def _init_attr(self):
        self.attr_name         = list(self.goods_attr) # ['price', 'size']
        self.attr_type         = [list(map(lambda x:name+'_'+x,level)) for name,level in self.goods_attr.items()] # [['price_l','price_s'], ['size_l','size_s']]
        self.attr_type_flatten = list(chain(*self.attr_type))
        self.goods_type        = list(map(lambda attrs:reduce(lambda x,y:x+'*'+y,attrs),product(*self.attr_type))) #['price_l * size_l', 'price_l * size_m']
    
    def _init_attr_trans(self):
        """
        统计数据中各属性的转移频数矩阵,当频数较低时,代表估计的参数可能会不准确
        """
        attr_count = len(self.attr_type)
        attr_trans = list()
        for attr in range(attr_count):
            attr_levels = self.attr_type[attr]
            df_list = list()
            for attr_level in attr_levels:
                # 出是否有空值
                trans_out = self.data.filter(like=attr_level)
                # 入是否无空值
                trans_in = self.data.drop(labels=trans_out.columns,axis=1)
                
                # 转出中任意一个为空值为1
                trans_out = trans_out.mask(~trans_out.isna(),0)
                trans_out = trans_out.mask(trans_out.isna(),1)
                trans_out = trans_out.sum(axis=1)
                trans_out = trans_out.mask(trans_out>0,1).reset_index(drop=True)
                trans_out.name = 'out'
                
                # 转入中任意一个不为空值为1
                trans_in = trans_in.mask(~trans_in.isna(),1)
                trans_in = trans_in.mask(trans_in.isna(),0)
                trans_in_col = list(trans_in.columns)
                trans_in_col = list(map(lambda x:x.split('*')[attr],trans_in_col))
                trans_in = trans_in.set_axis(trans_in_col,axis=1)
                trans_in = trans_in.reset_index(drop=True).stack().groupby(level=[0,1]).sum().unstack()
                trans_in = trans_in.mask(trans_in > 0, 1)
                
                df = pd.concat([trans_out,trans_in],axis=1)
                df = df.groupby('out').sum().reindex([0,1],axis=0).loc[lambda dt:dt.index==1]
                df_list.append(
                    df.set_axis([attr_level],axis=0).reindex(attr_levels,axis=1)
                )
            attr_trans.append(reduce(lambda x,y:pd.concat([x,y],axis=0),df_list))
        self.attr_trans = attr_trans
    
    def init_theta(self,theta_flatten,reparam=True,to_goods=True,goods_to_numpy=True):
        """
        初始化待估计的参数,将theta列表转化为商品选择概率向量和商品转移概率矩阵

        Parameters
        ----------
        theta_flatten : list
            待估计的参数,由两部分组成: 
                1.各属性的选择概率 
                2.各属性间的转移概率 
        reparam : bool, optional
            是否对所估计的参数reparametrizing, by default True
        to_goods : bool, optional
            是否将参数转换至商品维度, by default True

        Returns
        -------
        dict
            theta_f为商品的选择概率向量 
            theta_pi为商品间的转移概率矩阵 
        """
        # 避免优化的过程中theta_flatten变为ndarray
        theta_flatten = list(theta_flatten)
        
        # 从theta_flatten中提取相应部分的内容
        ## theta_flatten中涉及各属性的选择概率的索引
        theta_f_index = map(lambda x:len(x),self.attr_type)
        theta_f_index = [0]+list(accumulate(theta_f_index,func = lambda x,y:x+y))
        
        ## theta_flatten中涉及各属性间的转移概率的索引
        theta_pi_index = map(lambda x:len(x)**2-len(x),self.attr_type)
        theta_pi_index = list(accumulate(theta_pi_index,func = lambda x,y:x+y))
        theta_pi_index = [theta_f_index[-1]] + list(map(lambda x:x+theta_f_index[-1],theta_pi_index))
        
        ## 通过索引分割theta_flatten
        theta_f_list          = [] # 属性选择概率
        theta_pi_flatten_list = [] # 属性转移概率
        for i in range(len(theta_f_index)-1):
            theta_f = theta_flatten[theta_f_index[i]:theta_f_index[i+1]]
            theta_pi = theta_flatten[theta_pi_index[i]:theta_pi_index[i+1]]
            
            # 重参数 reparametrizing
            if reparam:
                theta_f = np.exp(theta_f) / np.sum(np.exp(theta_f))
                theta_pi = np.exp(theta_pi) / (1 + np.exp(theta_pi))
                
            theta_f_list.append(theta_f)
            theta_pi_flatten_list.append(theta_pi)
            
        ## 将属性转移概率转换为矩阵形式
        theta_pi_list = []
        for i in range(len(theta_pi_flatten_list)):
            dim = len(self.attr_type[i])  # 各属性下不同水平的个数
            pi = theta_pi_flatten_list[i]
            # 生成单位矩阵并填充非对角线元素
            m = np.identity(dim)
            pi_triu = pi[0:int(len(pi)/2)]
            pi_tril = pi[int(len(pi)/2):len(pi)]
            m[np.triu_indices(n=dim, k=1)] = pi_triu  # 上三角，从左到右从上到下填充
            m[np.tril_indices(n=dim, k=-1)] = pi_tril  # 下三角
            theta_pi_list.append(m)
        
        if to_goods:
            # 通过【属性】的选择概率和转移概率 转换为 【商品】的选择概率和转移概率
            ## 将属性概率笛卡尔积并相乘后得到商品的选择概率
            theta_f = list(map(lambda l:reduce(lambda x,y:x*y,l),product(*theta_f_list)))
            theta_f = pd.Series(data=theta_f,index=self.goods_type)
                
            ## 将属性转移概率转换为商品转移概率（使用克罗内克积）
            theta_pi = reduce(np.kron, theta_pi_list)
            theta_pi = pd.DataFrame(data=theta_pi, 
                                    columns=self.goods_type, 
                                    index=self.goods_type)
            
            if goods_to_numpy is True:
                theta_f  = theta_f.to_numpy()
                theta_pi = theta_pi.to_numpy()
            
            return theta_f,theta_pi
        else :
            #【属性】的选择概率和转移概率
            return theta_f_list,theta_pi_flatten_list         
    
    def get_loglikelihood(self,theta_flatten):
        """
        计算对数似然函数

        Parameters
        ----------
        theta_flatten : ndarray
            带估计的参数

        Returns
        -------
        float
            似然函数值
        """
        theta_f,theta_pi = self.init_theta(theta_flatten=theta_flatten)
        data = self.data.to_numpy()
        llh = loglikelihood(theta_f=theta_f,
                            theta_pi=theta_pi,
                            data_numpy=data)
        return llh
    
    # def single_likelihood(self,theta_f,theta_pi,sample):
    #     """
    #     当个样本的似然

    #     Parameters
    #     ----------
    #     theta : dict
    #         通过self.init_theta得到的theta
    #     sr : Pandas Series
    #         单个样本

    #     Returns
    #     -------
    #     float
    #         似然
    #     """
    #     theta_f = theta_f
    #     theta_pi = theta_pi
        
    #     trans_pi = theta_pi[np.isnan(sample),:]
    #     trans_pi[:,np.isnan(sample)] = 0
        
    #     # 转移矩阵只保留每行最大的元素
    #     trans_pi_max = np.zeros_like(trans_pi)
    #     row_indices = np.arange(trans_pi.shape[0])
    #     col_indices = np.argmax(trans_pi,axis=1)
    #     trans_pi_max[row_indices,col_indices] = trans_pi[row_indices,col_indices]
        
    #     trans_f = np.dot(theta_f[np.isnan(sample)],trans_pi_max)
        
    #     theta_f_total = theta_f + trans_f
    #     theta_f_total[np.isnan(sample)] = 0
    #     theta_f_total = theta_f_total[~np.isnan(sample)]
    #     sample = sample[~np.isnan(sample)] 
        
    #     llh = np.sum(sample*np.log(theta_f_total)) - np.sum(sample)*np.log(np.sum(theta_f_total))
    #     return llh
        
    # def likelihood(self,theta_flatten,reparam=True):
    #     """
    #     似然函数

    #     Parameters
    #     ----------
    #     theta_flatten : list
    #         待估计参数
    #     reparam : bool
    #         当设为False时,theta_flatten必须为百分比值,仅用于验证目的

    #     Returns
    #     -------
    #     float
    #         似然(取负值)
    #     """
        
    #     theta_f,theta_pi = self.init_theta(theta_flatten  = theta_flatten,
    #                                        reparam        = reparam,
    #                                        to_goods       = True,
    #                                        goods_to_numpy = True)
        
    #     llh_list = np.apply_along_axis(func1d=lambda x:self.single_likelihood(theta_f  = theta_f,
    #                                                                           theta_pi = theta_pi,
    #                                                                           sample   = x),
    #                                    axis=1,
    #                                    arr=self.data.to_numpy())
    #     return -np.sum(llh_list)
    
    def fit(self,method='differential_evolution',**kwargs):
        """
        模型拟合,估计参数

        Parameters
        ----------
        method : str, optional
            最优化方法,见scipy的globel optim, by default 'differential_evolution'

        Returns
        -------
        dict
            scipy的最优化结果
        """
        f_flatten_len = len(self.attr_type_flatten)
        pi_flatten_len = sum(list(map(lambda x:len(x)**2-len(x),self.attr_type)))
        
        bounds_up = [1]*f_flatten_len + [3]*pi_flatten_len
        bounds_dw = [-1]*f_flatten_len + [-3]*pi_flatten_len
        bounds = list(zip(bounds_dw,bounds_up))
        
        if method == 'differential_evolution':
            opt = differential_evolution(
                func=self.get_loglikelihood,
                x0=np.random.normal(loc=0,scale=0.1,size=f_flatten_len+pi_flatten_len),
                bounds=bounds,
                workers=-1,
                **kwargs)
            
        if method == 'basinhopping':
            opt = basinhopping(
                func=self.get_loglikelihood,
                x0=np.random.normal(loc=0,scale=0.1,size=f_flatten_len+pi_flatten_len),
                minimizer_kwargs={'method':'BFGS'},
                **kwargs)
        
        if method == 'dual_annealing':
            opt = dual_annealing(
                func= self.get_loglikelihood,
                bounds = bounds,
                **kwargs
            )
        
        self.theta_hat = opt
        
        return self.theta_hat
    
    def validation(self,SimSale,interval=0.1):
        """
        通过模拟数据严重参数的估计结果

        Parameters
        ----------
        SimSale : SimSale对象
            模拟的数据对象
        """
        x_goods_pi_theta     = SimSale.goods_pi.to_numpy().flatten()
        y_goods_pi_theta_hat = self.init_theta(self.theta_hat.x)[1].flatten()
        
        plt.scatter(x_goods_pi_theta[x_goods_pi_theta!=1],
                    y_goods_pi_theta_hat[y_goods_pi_theta_hat!=1])
        plt.axline([0,0],slope=1,color='r')
        plt.axline([0,-interval],slope=1,color='b',ls='--')
        plt.axline([0,interval],slope=1,color='b',ls='--')
        plt.xlabel('theta')
        plt.ylabel('theta_hat')
    
@jit(nopython = True)
def loglikelihood(theta_f,theta_pi,data_numpy):
    sum_llh = 0
    for sample in data_numpy:
        trans_pi = theta_pi[np.isnan(sample),:]
        trans_pi[:,np.isnan(sample)] = 0
        
        # 转移矩阵只保留每行最大的元素
        trans_pi_max = np.zeros_like(trans_pi)
        row_indices = np.arange(trans_pi.shape[0])
        col_indices = np.arange(trans_pi.shape[1])
        col_max_indices = np.argmax(trans_pi,axis=1)
        
        for i in row_indices:
            col_max_index = col_max_indices[i]
            for j in col_indices:
                if j == col_max_index:
                    trans_pi_max[i,j] = np.max(trans_pi[i])
                else :
                    pass
        
        trans_f = np.dot(theta_f[np.isnan(sample)],trans_pi_max)
        
        theta_f_total = theta_f + trans_f
        theta_f_total[np.isnan(sample)] = 0
        theta_f_total = theta_f_total[~np.isnan(sample)]
        sample = sample[~np.isnan(sample)] 
        
        llh = np.sum(sample*np.log(theta_f_total)) - np.sum(sample)*np.log(np.sum(theta_f_total))
        sum_llh += llh
    return -sum_llh
