# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:27:29 2019
rnn-lstm神经网络股票预测项目
tensorflow 1.13.1
@author: JL_zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

#配置matplotlib画图的符号
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示坐标中的负号



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
#normalize_data=(data-np.mean(data))/np.std(data)
#normalize_data=normalize_data[:,np.newaxis]   
#fig =plt.figure()
#plt.plot(data[:,1])

#———————————————————形成训练集—————————————————————
#设置rnn网络的常量
time_step=1     #时间步 ，rnn每迭代20次，就向前推进一步
rnn_unit=128       # hidden layer units
batch_size=400     # 每一批训练多少个样例
input_size=6     # 输入层数维度
output_size=1     # 输出层数维度
lr=0.00001         # 学习率
layer_num = 7
day_num = 5
#tf.reset_default_graph()
#—————————————————————获取训练集——————————————————
def get_train_data(data,day_num,batch_size,time_step,train_begin=0,train_end=2050):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集x和y初定义
    for i in range(len(normalized_train_data)-time_step-day_num):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:6]
       y=normalized_train_data[i + day_num:i+time_step + day_num,1,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y
"""
#——————————————————————获取测试集——————————————————————
def get_test_data(time_step=5,test_begin=1800):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_test_data[i*time_step:(i+1)*time_step,1]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:4]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,1]).tolist())
    return mean,std,test_x,test_y

"""
#———————————————————定义神经网络变量—————————————————————
tf.reset_default_graph()

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
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=0.6)
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



#———————————————————对模型进行训练—————————————————————
def train_lstm(data,day_num,batch_size,time_step,train_begin=0,train_end=2050):
    Loss_fun = []
    #X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
    #Y=tf.placeholder(tf.float32, [None,time_step,output_size])   # 每批次tensor对应的标签
    batch_index,train_x,train_y=get_train_data(data,day_num,batch_size,time_step,train_begin,train_end)
    #global batch_size
   # with tf.variable_scope("sec_lstm", reuse=True):
    pred,_=lstm(batch_size,rnn_unit,layer_num)
    #定义损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    #checkpoint_dir='G:/Stock_predict/model'
    mod_name = str(day_num) + 'modle.ckpt'
    checkpoint_dir='G:/Stock_predict/model1' + str(day_num) + '/'
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(1000): #We can increase the number of iterations to gain better result.
    
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #每训练100次保存一次参数
                if step%100==0:
                    #print("Number of iterations:",i," loss:",loss_) #输出训练次数，输出损失值
                    #print("model_save",saver.save(sess,checkpoint_dir+'modle.ckpt')) #第二个参数是保存的地址，可以修改为自己本地的保存地址
                    saver.save(sess,checkpoint_dir+mod_name)
                    #I run the code in windows 10,so use  'model_save1\\modle.ckpt'
                    #if you run it in Linux,please use  'model_save1/modle.ckpt'
                print("Number of iterations:",i,"The step is: ", step," loss:",loss_) #输出训练次数，输出损失值
                Loss_fun.append(loss_)
                step+=1
        #saver.save(sess,'G:/Stock_predict/model/modle.ckpt')
        print("The train has finished")
        plt.plot(list(range(len(Loss_fun))),Loss_fun)
train_lstm(data,day_num,batch_size,time_step)
# #对模型进行训练
"""
#———————————————————预测模型—————————————————————
def prediction(time_step,X,weights,biases):
    mean,std,test_x,test_y=get_test_data(time_step,test_begin=1800)
    checkpoint_dir='G:/Stock_predict/model'
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(1,X,weights,biases,rnn_unit,layer_num)    #预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3) 
    with tf.Session() as sess:
        #参数恢复
        ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path) #第二个参数是保存的地址，可以修改为自己本地的保存地址
        #I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        #if you run it in Linux,please use  'model_save1/modle.ckpt'
        
        #取训练集最后一行为测试样本。shape = [1,time_step,input_size]
        module_file = tf.train.latest_checkpoint()
        saver.restore(sess, module_file) 
        prev_seq=test_x[-1]
        predict=[]
        #得到之后的100个预测结果
        for i in range(100):  #预测100个数值
            next_seq=sess.run(pred,feed_dict={X:[test_x[time_step]]})
            predict.append(next_seq[-1])
            #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试数据
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
        #以折线图展示结果
        plt.figure(figsize = (8,8)) #图像大小为8*8英寸
        #设置背景风格
        sns.set_style(style = 'whitegrid') #详细参数看seaborn的API  http://seaborn.pydata.org/api.html
        #设置字体
        sns.set_context(context = 'poster',font_scale = 1)
        plt.title('lstm_rnn_stock_prediction')
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b',label = 'raw data') #这是原来股票的价格走势，用蓝色曲线表示
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r',label = 'predict trend') #预测未来的价格走势用红色表示
        plt.legend(loc = 'best')
        plt.xticks([])#去掉X轴刻度
        plt.xticks([])
        plt.show()
        
prediction(time_step,X,weights,biases) 
"""


























