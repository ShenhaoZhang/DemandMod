from itertools import product,accumulate,chain
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from numba import jit

class Demand:
    """
    需求估计模型
    """
    def __init__(self, data:pd.DataFrame, numba:bool=True) -> None:
        self.data       = data.sort_index(axis=1)
        self.numba      = numba
        
        self.goods_attr             = None # {'price': ['l', 'm', 's'], 'size': ['l', 'm', 's']}
        self.attr_name              = None # ['price', 'size']
        self.attr_type              = None # [['price_l','price_s'], ['size_l','size_s']]
        self.attr_type_flatten      = None # ['price_l','price_s','size_l','size_s']
        self.goods_type             = None # ['price_l * size_l', 'price_l * size_m']
        self.index_flatten_theta_f  = None
        self.index_flatten_theta_pi = None
        self.attr_trans             = None
        
        self._init_attr()
        self._init_attr_trans()
        
        self.fit_result         = None # scipy中的优化结果
        self.theta_hat_attr_f   = None # 属性的选择概率
        self.theta_hat_attr_pi  = None # 属性的转移概率
        self.theta_hat_goods_f  = None # 商品的选择概率
        self.theta_hat_goods_pi = None # 商品的转移概率
        
    def _init_attr(self):
        """
        初始化属性
        """
        # 从数据中初始化商品的属性
        attr_lvl = [attr_lvl for attr_lvl_list in self.data.columns.str.split('*').to_list() for attr_lvl in attr_lvl_list]
        attr_lvl = list(set(attr_lvl))
        attr_lvl_df = pd.Series(attr_lvl).str.split('_',expand=True).set_axis(['attr','lvl'],axis=1).sort_values(['attr','lvl'])
        self.goods_attr = attr_lvl_df.groupby('attr').agg(list).loc[:,'lvl'].to_dict()
        
        # 由商品属性衍生出的属性
        self.attr_name         = list(self.goods_attr)
        self.attr_type         = [list(map(lambda x:name+'_'+x,level)) for name,level in self.goods_attr.items()] 
        self.attr_type_flatten = list(chain(*self.attr_type))
        self.goods_type        = list(map(lambda attrs:reduce(lambda x,y:x+'*'+y,attrs),product(*self.attr_type)))
        
        # theta_flatten中涉及各属性的选择概率的索引，如[0,2,4]
        self.index_flatten_theta_f = map(lambda x:len(x),self.attr_type)
        self.index_flatten_theta_f = [0]+list(accumulate(self.index_flatten_theta_f,func = lambda x,y:x+y))
        # theta_flatten中涉及各属性间的转移概率的索引,如[4,6,8]
        self.index_flatten_theta_pi = map(lambda x:len(x)**2-len(x),self.attr_type)
        self.index_flatten_theta_pi = list(accumulate(self.index_flatten_theta_pi,func = lambda x,y:x+y))
        self.index_flatten_theta_pi = [self.index_flatten_theta_f[-1]] + list(map(lambda x:x+self.index_flatten_theta_f[-1],self.index_flatten_theta_pi))
    
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
                trans_out      = trans_out.mask(~trans_out.isna(),0)
                trans_out      = trans_out.mask(trans_out.isna(),1)
                trans_out      = trans_out.sum(axis=1)
                trans_out      = trans_out.mask(trans_out>0,1).reset_index(drop=True)
                trans_out.name = 'out'
                
                # 转入中任意一个不为空值为1
                trans_in     = trans_in.mask(~trans_in.isna(),1)
                trans_in     = trans_in.mask(trans_in.isna(),0)
                trans_in_col = list(trans_in.columns)
                trans_in_col = list(map(lambda x:x.split('*')[attr],trans_in_col))
                trans_in     = trans_in.set_axis(trans_in_col,axis=1)
                trans_in     = trans_in.reset_index(drop=True).stack().groupby(level=[0,1]).sum().unstack()
                trans_in     = trans_in.mask(trans_in > 0, 1)
                
                df = pd.concat([trans_out,trans_in],axis=1)
                df = df.groupby('out').sum().reindex([0,1],axis=0).loc[lambda dt:dt.index==1]
                df_list.append(
                    df.set_axis([attr_level],axis=0).reindex(attr_levels,axis=1)
                )
            attr_trans.append(reduce(lambda x,y:pd.concat([x,y],axis=0),df_list))
        self.attr_trans = attr_trans
    
    def init_theta(self,theta_flatten,reparam=True,attrs_to_Pandas=False,to_goods=True,goods_to_Pandas=False):
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
        
        ## 通过索引分割theta_flatten
        theta_f_list          = [] # 属性选择概率
        theta_pi_flatten_list = [] # 属性转移概率
        for i in range(len(self.index_flatten_theta_f)-1):
            theta_f = theta_flatten[self.index_flatten_theta_f[i]:self.index_flatten_theta_f[i+1]]
            theta_pi = theta_flatten[self.index_flatten_theta_pi[i]:self.index_flatten_theta_pi[i+1]]
            # 重参数 reparametrizing
            if reparam is True:
                theta_f = np.exp(theta_f) / np.sum(np.exp(theta_f))
                # theta_pi = np.exp(theta_pi) / (1 + np.exp(theta_pi))
            if attrs_to_Pandas is True:
                theta_f = pd.Series(data=theta_f,index=self.attr_type[i])
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
        
        if to_goods is True:
            # 通过【属性】的选择概率和转移概率 转换为 【商品】的选择概率和转移概率
            ## 将属性概率笛卡尔积并相乘后得到商品的选择概率
            theta_f = np.array(list(map(lambda l:reduce(lambda x,y:x*y,l),product(*theta_f_list))))
            ## 将属性转移概率转换为商品转移概率（使用克罗内克积）
            theta_pi = reduce(np.kron, theta_pi_list)
            
            ## 将选择概率和转移概率以Pandas数据结构展示，用于展示，不实际计算
            if goods_to_Pandas is True:
                theta_f  = pd.Series(data=theta_f,index=self.goods_type)
                theta_pi = pd.DataFrame(data=theta_pi, columns=self.goods_type, index=self.goods_type)
            
            return theta_f,theta_pi
        
        else :
            #【属性】的选择概率和转移概率
            return theta_f_list,theta_pi_flatten_list
    
    @staticmethod
    @jit(nopython = True)
    def loglikelihood_numba(theta_f,theta_pi,data_numpy):
        """
        计算似然函数值, 通过numba加速

        Parameters
        ----------
        theta_f : ndarray
            直接选择某个商品的概率
        theta_pi : ndarray
            属性之间的转移概率
        data_numpy : ndarray
            样本数据

        Returns
        -------
        float
            似然函数值
        """
        
        sum_llh = 0
        for sample in data_numpy:
            # 构造数据对应的转移矩阵，只保留存在缺失商品的行，缺失商品的列设为0
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
        
        theta_f,theta_pi = self.init_theta(theta_flatten = theta_flatten)
        data = self.data.to_numpy()
        
        # 两种途径计算似然函数，numba计算效率高，python适应性强
        if self.numba is True:
            loglikelihood = Demand.loglikelihood_numba
        else:
            pass
            # loglikelihood = loglikelihood_python
            
        llh = loglikelihood(theta_f = theta_f, theta_pi = theta_pi, data_numpy = data)
        
        return llh
    
    def fit(self,method='dual_annealing',**kwargs):
        """
        模型拟合,估计参数

        Parameters
        ----------
        method : str, optional
            最优化方法,见scipy的globel optim, by default 'dual_annealing'

        Returns
        -------
        dict
            scipy的最优化结果
        """
        
        f_flatten_len     = len(self.attr_type_flatten)
        pi_flatten_len    = sum(list(map(lambda x:len(x)**2-len(x),self.attr_type)))
        theta_flatten_len = f_flatten_len + pi_flatten_len
        # 边界
        bounds_up = [1]*f_flatten_len + [0.99]*pi_flatten_len
        bounds_dw = [-1]*f_flatten_len + [0.01]*pi_flatten_len
        bounds    = list(zip(bounds_dw,bounds_up))
        # 初始值
        x0_theta_hat_f  = [f for f_list in map(lambda x:x.to_list(),self.guess_theta_f()) for f in f_list]
        x0_theta_hat_pi = np.zeros(shape=pi_flatten_len)+0.1
        x0              = np.append(x0_theta_hat_f,x0_theta_hat_pi)
        
        if method == 'differential_evolution':
            opt = optimize.differential_evolution(
                func    = self.get_loglikelihood,
                x0      = x0,
                bounds  = bounds,
                workers = -1,
                **kwargs)
        
        elif method == 'dual_annealing':
            opt = optimize.dual_annealing(
                func   = self.get_loglikelihood,
                bounds = bounds,
                **kwargs
            )
        else:
            opt = optimize.minimize(
                fun    = self.get_loglikelihood,
                x0     = x0,
                bounds = bounds,
                **kwargs
            )
        
        self.fit_result         = opt
        self.theta_hat_attr_f   = self.init_theta(opt['x'],to_goods=False,reparam=True)[0]
        self.theta_hat_attr_pi  = self.init_theta(opt['x'],to_goods=False,reparam=True)[1]
        self.theta_hat_goods_f  = self.init_theta(opt['x'],to_goods=True,reparam=True)[0]
        self.theta_hat_goods_pi = self.init_theta(opt['x'],to_goods=True,reparam=True)[1]
        
    def fit_skopt(self,**kwargs):
        """
        模型拟合,估计参数

        Parameters
        ----------
        method : str, optional
            最优化方法,见scipy的globel optim, by default 'dual_annealing'

        Returns
        -------
        dict
            scipy的最优化结果
        """
        from skopt.optimizer import gp_minimize
        from skopt.space import Real
        
        f_flatten_len = len(self.attr_type_flatten)
        pi_flatten_len = sum(list(map(lambda x:len(x)**2-len(x),self.attr_type)))
        
        space = []
        for i in range(f_flatten_len+pi_flatten_len):
            if i + 1 <= f_flatten_len:
                space.append(Real(-1,1))
            else:
                space.append(Real(0,1))
        
        opt = gp_minimize(self.get_loglikelihood,space,**kwargs)
        self.fit_result         = opt
        self.theta_hat_attr_f   = self.init_theta(opt['x'],to_goods=False,reparam=True)[0]
        self.theta_hat_attr_pi  = self.init_theta(opt['x'],to_goods=False,reparam=True)[1]
        self.theta_hat_goods_f  = self.init_theta(opt['x'],to_goods=True,reparam=True)[0]
        self.theta_hat_goods_pi = self.init_theta(opt['x'],to_goods=True,reparam=True)[1]
        
        return opt['x']
    
    def summary(self,SimSale=None):
        """
        模型的总结

        Parameters
        ----------
        SimSale : SimSale对象, optional
            模拟的销售数据, by default None
        """
        if self.fit_result is None:
            raise Exception('Model Need Fit')
        print('#'*20+' 1.属性的选择概率 '+'#'*20)
        attrs_f_list = self.init_theta(self.fit_result['x'],attrs_to_Pandas=True,to_goods=False,reparam=True)[0]
        attrs_f_list_guess = self.guess_theta_f()
        for attr_index in range(len(attrs_f_list)):
            attrs_df = (
                pd.concat([attrs_f_list[attr_index],attrs_f_list_guess[attr_index]],axis=1)
                .set_axis(['Estimate','Guess'],axis=1)
                .assign(EG_Diff = lambda dt:dt.Estimate - dt.Guess))
            if SimSale is not None:
                attrs_df = attrs_df.assign(
                    Real = lambda dt:SimSale.attr_f_list[attr_index].to_numpy(),
                    ER_Diff = lambda dt:dt.Estimate - dt.Real,
                    GR_Diff = lambda dt:dt.Guess - dt.Real)
            print(attrs_df)

        if SimSale is not None:
            print('#'*20+' 2.属性和商品的转移概率 '+'#'*20)
            self.score(SimSale=SimSale,plot=True)
        
    def predict(self,data:pd.DataFrame):
        """
        通过拟合得到的选择概率及转移概率，预测商品如果在售时的销量

        Parameters
        ----------
        data : pd.DataFrame
            实际的销售数据

        Returns
        -------
        pd.DataFrame
            预测的销售数据

        """
        #TODO 预测全量数据
        if self.fit_result is None:
            raise Exception('Model Need Fit')
        # 数据验证
        ## 验证是否存在需要预测的值
        if data.isna().any(axis=1).to_numpy()[0] is False:
            return data
        data = data.sort_index(axis=1)
        ## 验证行数和列数
        n_row,n_col = data.shape
        if (n_row != 1) or (n_col != len(self.goods_type)):
            raise Exception('Wrong Data Shape')
        ## 验证列名是否相同
        if data.columns.to_list() != self.goods_type:
            raise Exception('Wrong Data Columns')
        
        # 估计如果所有商品类型都在售的情况下的总销量
        ## 在售品的索引及未售品的索引
        data_sr = data.squeeze()
        index_unsale = data_sr.isna().to_list()
        index_sale   = data_sr.notna().to_list()
        goods_f_unsale = self.theta_hat_goods_f[index_unsale]
        goods_f_sale = self.theta_hat_goods_f[index_sale]
        goods_pi_unsale_to_sale = self.theta_hat_goods_pi[index_unsale][:,index_sale]
        goods_pi_unsale_to_sale = self.get_trans_prob(goods_pi_unsale_to_sale)
        total_sale = data_sr.to_numpy()[index_sale] / (goods_f_sale + np.dot(goods_f_unsale , goods_pi_unsale_to_sale))
        total_sale = np.mean(total_sale)
        
        predict_sale = pd.DataFrame(data=dict(zip(self.goods_type,total_sale*self.theta_hat_goods_f)),index=[0])
        return predict_sale.round().astype('int')
    
    def conf_int(self,method='Bootstrap',level=0.95,plot=True,bootstrap_n=10,SimSale=None):
        """
        参数的置信区间

        Parameters
        ----------
        method : str
            计算置信区间的方法
        level : float, optional
            置信水平, by default 0.95
        plot : bool, optional
            返回图形, by default True
        bootstrap_n : int, optional
            Bootstrap的抽烟次数, by default 10
        SimSale : _type_, optional
            实例化的SimSale对象, by default None

        Returns
        -------
        tuple
            置信区间的上界和下界
        """
        
        # TODO 使用并行化计算，考虑从Method中移出，单独写成函数
        # 保存对象原始属性
        org_fit_result         = self.fit_result
        org_theta_hat_attr_f   = self.theta_hat_attr_f
        org_theta_hat_attr_pi  = self.theta_hat_attr_pi
        org_theta_hat_goods_f  = self.theta_hat_goods_f
        org_theta_hat_goods_pi = self.theta_hat_goods_pi
        
        if method == 'Bootstrap':
            theta_list = []
            for i in range(bootstrap_n):
                sample_data = self.data.sample(frac=1,replace=True)
                self.__init__(data=sample_data,goods_attr=self.goods_attr)
                theta_list.append(self.fit().x)
                
            theta_matrix = np.column_stack(theta_list)
            
            int_up       = np.quantile(theta_matrix,q=level+(1-level)/2,axis=1)
            int_up       = self.init_theta(int_up,to_goods=False)
            int_up       = np.append(np.concatenate(int_up[0]),np.concatenate(int_up[1]))
            
            int_dw       = np.quantile(theta_matrix,q=(1-level)/2,axis=1)
            int_dw       = self.init_theta(int_dw,to_goods=False)
            int_dw       = np.append(np.concatenate(int_dw[0]),np.concatenate(int_dw[1]))

            int_md       = np.quantile(theta_matrix,q=0.5,axis=1)
            int_md       = self.init_theta(int_md,to_goods=False)
            int_md       = np.append(np.concatenate(int_md[0]),np.concatenate(int_md[1]))
            
        if method == 'Wald':
            pass
        
        # 回复对象原始属性
        self.fit_result         = org_fit_result
        self.theta_hat_attr_f   = org_theta_hat_attr_f
        self.theta_hat_attr_pi  = org_theta_hat_attr_pi
        self.theta_hat_goods_f  = org_theta_hat_goods_f
        self.theta_hat_goods_pi = org_theta_hat_goods_pi
        
        if plot is True:
            fig,ax0 = plt.subplots()
            ax0.errorbar(
                x    = int_md,
                y    = np.arange(len(int_up)),
                xerr = [int_md - int_dw, int_up - int_md],
                fmt='o'
            )
            if SimSale is not None:
                ax0.scatter(x     = np.append(SimSale.attr_f_theta,SimSale.attr_pi_theta),
                            y     = np.arange(len(int_up)),
                            color = 'red')
        
        return int_up,int_dw,theta_list
    
    def score(self,SimSale,interval=0.1,plot=False,plot_ci=False):
        """
        通过模拟数据验证参数的估计结果

        Parameters
        ----------
        SimSale : SimSale对象
            模拟的数据对象
        """
        
        # 属性转移概率参数
        x_attr_pi_theta     = SimSale.attr_pi_theta
        y_attr_pi_theta_hat = [pi for pi_list in self.theta_hat_attr_pi for pi in pi_list]
        
        # 商品转移概率参数
        x_goods_pi_theta     = SimSale.goods_pi.to_numpy().flatten()
        y_goods_pi_theta_hat = self.theta_hat_goods_pi.flatten()
        
        SSR = np.sum((x_goods_pi_theta-y_goods_pi_theta_hat)**2)
        SST = np.sum((y_goods_pi_theta_hat-np.mean(y_goods_pi_theta_hat))**2)
        R2  = 1 - SSR / SST
        
        if plot is True:
            
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
            
            ax1.scatter(x_attr_pi_theta,y_attr_pi_theta_hat)
            ax1.axline([0,0],slope=1,color='r')
            ax1.axline([0,-interval],slope=1,color='b',ls='--')
            ax1.axline([0,interval],slope=1,color='b',ls='--')
            ax1.set_xlabel('attr_pi')
            ax1.set_ylabel('attr_pi_hat')
            
            ax2.scatter(x_goods_pi_theta[x_goods_pi_theta!=1],
                        y_goods_pi_theta_hat[y_goods_pi_theta_hat!=1])
            ax2.axline([0,0],slope=1,color='r')
            ax2.axline([0,-interval],slope=1,color='b',ls='--')
            ax2.axline([0,interval],slope=1,color='b',ls='--')
            ax2.set_xlabel('goods_pi')
            ax2.set_ylabel('goods_pi_hat')
            ax2.set_title('R Square = '+ np.str(np.round(R2,2)))
        else:
            return f'score:{R2}'
    
    def guess_theta_f(self):
        guess_theta_f_list = []
        for attr_index in range(len(self.attr_type)):
            data = self.data.copy()
            data.columns = data.columns.str.split('*').map(lambda x:x[attr_index])
            attr_sale = data.transpose().groupby(by=lambda x:x).agg(np.sum).transpose().sum()
            attr_sale_pct = attr_sale / attr_sale.sum()
            guess_theta_f_list.append(attr_sale_pct)
        return guess_theta_f_list
    
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

