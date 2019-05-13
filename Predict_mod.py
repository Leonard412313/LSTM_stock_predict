# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:59:09 2019

@author: JL_zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


path = 'G:/Stock_predict/'
stock_ID = 'beixinluiqiao.csv'


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
        
data, date = get_data(path,stock_ID)
normalize_data=(data-np.mean(data))/np.std(data)
#normalize_data=normalize_data[:,np.newaxis]   
#fig =plt.figure()
#plt.plot(data[:,1])

#ori_data = data[1800:,1]
#plt.plot(list(range(len(ori_data))), ori_data, color='b',label = 'raw data')

#———————————————————形成训练集—————————————————————
time_step=1     #时间步 ，rnn每迭代20次，就向前推进一步
rnn_unit=128       # hidden layer units
batch_size=400     # 每一批训练多少个样例
input_size=6     # 输入层数维度
output_size=1     # 输出层数维度
lr=0.00001         # 学习率
layer_num = 7
day_num = 5
tf.reset_default_graph()
"""
#—————————————————————获取训练集——————————————————
def get_train_data(data,day_num,batch_size,time_step,train_begin=0,train_end=1800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集x和y初定义
    for i in range(len(normalized_train_data)-time_step-day_num):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:6]
       y=normalized_train_data[i + 1:i+time_step + 1,1,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y
    """
#——————————————————————获取测试集——————————————————————
def get_test_data(day_num,time_step,test_begin=2068):
    data_test = data[test_begin:]
    data_test2 = data[test_begin+day_num:]
    mean=np.mean(data_test,axis=0)
    mean2=np.mean(data_test2,axis=0)
    std=np.std(data_test,axis=0)
    std2=np.std(data_test2,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    normalized_test_data2=(data_test2-mean2)/std2  #标准化
    size=(len(normalized_test_data))//time_step  #有size个sample 
    test_x,test_y=[],[]  
    for i in range(size):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_test_data2[(i)*time_step:(i+1)*time_step,1]
       test_x.append(x.tolist())
       test_y.extend(y)
    #test_x.append((normalized_test_data[(i+1)*time_step:,:6]).tolist())
    #test_y.extend((normalized_test_data[(i+1)*time_step:,1]).tolist())
    return mean,std,test_x,test_y


#———————————————————定义神经网络变量—————————————————————
#tf.reset_default_graph()

X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   # 每批次tensor对应的标签
#keep_prob = tf.placeholder(tf.float32)
#输入层、输出层的权重和偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }
#———————————————————定义lstm网络—————————————————————

def lstm(batch,rnn_unit,layer_num):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转为2维进行计算，计算后的结果作为 隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])   #将tensor转为3维，作为 lstm cell的输入
    #lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_unit, forget_bias=1.0, state_is_tuple=True)
    def get_a_cell():
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_unit, forget_bias=1.0, state_is_tuple=True, reuse=tf.AUTO_REUSE)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=0.8, output_keep_prob=0.8)
        return cell
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(layer_num)])
    
    #lstm_cell=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    #mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
    init_state=mlstm_cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(mlstm_cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit])  #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


#———————————————————预测模型—————————————————————
def prediction(time_step,rnn_unit,layer_num,day_num):
    mean,std,test_x,test_y=get_test_data(day_num,time_step,test_begin=2070)
    
   # _,train_x,_ = get_train_data(data,batch_size=30,time_step=5,train_begin=0,train_end=1800)
    #checkpoint_dir='G:/Stock_predict/model'
    checkpoint_dir='G:/Stock_predict/model1' + str(day_num) + '/'
    #with tf.variable_scope("sec_lstm",reuse=True):
    pred,_=lstm(1,rnn_unit,layer_num)    #预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.trainable_variables()) 
    with tf.Session() as sess:
        #参数恢复
        ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path) #第二个参数是保存的地址，可以修改为自己本地的保存地址
        #I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        #if you run it in Linux,please use  'model_save1/modle.ckpt'
        
        #取训练集最后一行为测试样本。shape = [1,time_step,input_size]
        module_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, module_file) 
        test_predict=[]
        for step in range(len(test_x)):
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})
            #print(prob)
            predict=prob.reshape((-1))
            test_predict.extend(predict)
        #print(len(test_predict))
        #print(test_predict)
        #print(len(predict))
        #print(len(test_predict))
        #print(step)
        test_y=np.array(test_y)*std[1]+mean[1]
        test_predict=np.array(test_predict)*std[1]+mean[1]
        acc=np.average(np.abs(test_predict[0:len(test_predict)-day_num]-test_y)/test_y) #acc为测试集偏差
        print("The bias is ", acc )
        sns.set_style(style = 'whitegrid')
        plt.title('lstm_rnn_stock_prediction')
        plt.figure(figsize = (14,14))
        plt.figure()
        plt.plot(list(range(0,len(test_predict))), test_predict, color='r',label = 'Predict data')
        plt.plot(list(range(len(test_y))), test_y,  color='b',label = 'Real data')
        plt.legend(loc = 'best')
        plt.xticks([])
        plt.show()

        return test_predict,test_y
        
newday,_ = prediction(time_step,rnn_unit,layer_num,day_num) 
