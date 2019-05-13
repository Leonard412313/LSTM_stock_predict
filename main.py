# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:36:04 2019



@author: JL_zhang
"""

import pandas as pd
import Train_mod as Train_mod
import tensorflow as tf
#import Predict_mod as Predict_mod

#data = ts.get_hist_data('002575', '2019-12-03', '2019-03-27')
#——————————————————设置常量——————————————————————
time_step=15     #时间步 ，rnn每迭代20次，就向前推进一步
rnn_unit=4       # 隐藏层节点数
batch_size=20     # 每一批训练多少个样例
input_size=6     # 输入层数维度
output_size=1     # 输出层数维度
lr=0.0006         # 学习率
layer_num = 4   #隐藏层层数
day_num = 5   #预测未来一周的走势
path = 'G:/Stock_predict/'
stock_ID = 'Qunxingwanjv222.csv'



#tf.reset_default_graph()
def get_data(path, stock_ID):
    try:
        data = pd.read_csv(path + stock_ID)
        date = data['date']
        data = pd.DataFrame(data)
        data = data.iloc[:,1:7].values        
        print('股票数据获取完成！')
        return data, date
    except Exception:
        print('股票读取失败！')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    