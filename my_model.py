# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:35:46 2019

@author: yangxiaokang
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

class DNN():
    
    def __init__(self,hidden_nodes=32,input_dim=1,lr= 0.1, activation=None, optimizer='sgd'):
        self.input_dim = input_dim        
        self.hidden_nodes = hidden_nodes
        self.lr = lr
        self.activation = activation
        self.optimizer=optimizer
        #self.build_DNN()
       
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
 
    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def tanh(self,x):
        return 2.0/(1+np.exp(-2*x))-1.0
 
    def tanh_derivative(self,x):
        return 1-self.tanh(x)*self.tanh(x)

    def relu(self,x):
        return np.where(x>0,x,0.01*x)
 
    def relu_derivative(self,x):
        return np.where(x>0,1,0.1)
        
    def sgd(self, dw,db):
        """
        单纯的sgd实现
        """
        next_w = self.w - self.lr*dw
        next_b = self.b - self.lr*db
        return next_w, next_b

    def adagrad(self, dw, db):
        
        self.cache_w =0.99*self.cache_w + (1-0.99)*(dw**2)
        self.cache_b =0.99*self.cache_b + (1-0.99)*(db**2)
        #self.cache_w += dw**2
        #self.cache_b += db**2
        next_w = self.w - self.lr*dw / (np.sqrt(self.cache_w) + 1e-7)
        next_b = self.b - self.lr*db / (np.sqrt(self.cache_b) + 1e-7)
        return next_w, next_b
        
    def adam(self, dw, db):
        self.cache_w_mean = 0.9*self.cache_w_mean + (1-0.9)*dw
        self.cache_b_mean = 0.9*self.cache_b_mean + (1-0.9)*db
        self.cache_w_square = 0.999*self.cache_w_square + (1-0.999)*(dw**2)
        self.cache_b_square = 0.999*self.cache_b_square + (1-0.999)*(db**2)
        self.cache_t += 1
        mean_w = self.cache_w_mean / (1 - (0.9**self.cache_t))
        square_w = self.cache_w_square / (1 - (0.999**self.cache_t))
        
        mean_b = self.cache_b_mean / (1 - (0.9**self.cache_t))
        square_b = self.cache_b_square / (1 - (0.999**self.cache_t))
        
        next_w = self.w - self.lr*mean_w / (np.sqrt(square_w) + 1e-7)
        next_b = self.b - self.lr*mean_b / (np.sqrt(square_b) + 1e-7)        
        return next_w, next_b

    def build_DNN(self):
        np.random.seed(1)
        # Layer parameters
        self.w = np.random.normal(0.0,0.1,size=(self.hidden_nodes,self.input_dim))
        self.b = np.zeros(shape=(self.hidden_nodes,1))
        if self.optimizer=='adagrad':
            self.cache_w = np.zeros_like(self.w)
            self.cache_b = np.zeros_like(self.b)
        if self.optimizer=='adam':
            self.cache_w_mean = np.zeros_like(self.w)
            self.cache_w_square = np.zeros_like(self.w)
            self.cache_b_mean = np.zeros_like(self.b)
            self.cache_b_square = np.zeros_like(self.b)
            self.cache_t = 0
            
    def forwardPropagation(self,inputs):
        self.x = np.matmul(self.w,inputs) + self.b
        if self.activation==None:
            self.activation_out = self.x
        if self.activation=="sigmoid":
            self.activation_out = self.sigmoid(self.x)
        if self.activation=="relu":
            self.activation_out = self.relu(self.x)
        if self.activation=="tanh":
            self.activation_out = self.tanh(self.x)
        #return self.activation_out
        
    def backwardPropagation(self,da,a_prev,last=False):
        
        if last or self.activation==None:
            dx = da
        else:
            if self.activation=="sigmoid":
                dx = self.sigmoid_derivative(self.x)*da
            if self.activation=="relu":
                dx = self.relu_derivative(self.x)*da
            if self.activation=="tanh":
                dx = self.tanh_derivative(self.x)*da

        nums = da.shape[1]
        #dw = dx/dw(已知等号前的w1,w2,,导数式子和等号右边的梯度值，便可求出dw) 
        dw = np.matmul(dx,a_prev.T)/nums
        db = np.mean(dx,axis=1,keepdims=True)
        a_prev = np.matmul(self.w.T,dx)
    
        if self.optimizer=="sgd":
            self.w, self.b = self.sgd(dw,db)

        if self.optimizer=="adagrad":
            self.w, self.b = self.adagrad(dw,db)     
        if self.optimizer=="adam":
            self.w, self.b = self.adam(dw,db)
        return a_prev
    
class Model(object):
    
    def __init__(self):
        self.DNN_list = []
         
    def add(self,dnn_param):
        self.DNN_list.append(dnn_param)
        
    def compile(self, optimizer='sgd', loss=None, lr=0.1):
        for i in range(len(self.DNN_list)):
            if i!=0:     
                self.DNN_list[i].input_dim = self.DNN_list[i-1].hidden_nodes
            self.DNN_list[i].optimizer = optimizer
            self.DNN_list[i].build_DNN()
            self.DNN_list[i].lr = lr
                
    def fit(self,x=None,y=None,batch_size=None,epochs=1,**kwargs):
        loss_list = []
        for i in range(epochs):
            #print(epochs,i)
            for fp in range(len(self.DNN_list)):
                #print('fp ',fp)
                if fp==0:
                    self.DNN_list[fp].forwardPropagation(x)
                else:
                    self.DNN_list[fp].forwardPropagation(self.DNN_list[fp-1].activation_out)
            
            #loss = 0.5*np.mean((self.DNN_list[fp].activation_out-y)**2-0.02*np.log(np.fabs(self.DNN_list[fp].activation_out)))
            loss = 0.5*np.mean((self.DNN_list[fp].activation_out-y)**2)
            loss_list.append(loss)
            da = self.DNN_list[fp].activation_out - y
            #a_prev = self.DNN_list[fp-1].activation_out
            for bp in range(len(self.DNN_list)-1):
                #print('bp ',bp)
                a_prev = self.DNN_list[fp-1-bp].activation_out
                da = self.DNN_list[len(self.DNN_list)-bp-1].backwardPropagation(da, a_prev)
            self.view_bar(i+1,epochs,loss)
        return loss_list
        
    def predict(self,x=None):
        for fp in range(len( self.DNN_list)):
            if fp==0:
                 self.DNN_list[fp].forwardPropagation(x)
            else:
                 self.DNN_list[fp].forwardPropagation( self.DNN_list[fp-1].activation_out)
        return  self.DNN_list[fp].activation_out
            
    def view_bar(self,step,total,loss):
        rate = step/total
        rate_num = int(rate*40)
        r = '\repochs: %d loss value: %.4f[%s%s]\t%d%%  %d/%d'%(step,loss,'>'*rate_num,'-'*(40-rate_num),
                                      int(rate*100),step,total)
        sys.stdout.write(r)
        sys.stdout.flush()
        time.sleep(0.0001)

def norm(x):
    """"
    归一化到区间[0.01,0.99]
    """
    _range = np.max(x) - np.min(x)
    k = (1.-0.)/_range
    return 0.01 + k*(x - np.min(x))
            
if __name__ == '__main__':
    length = 2000
    #arr = np.arange(length)
    #np.random.shuffle(arr)
    X = np.linspace(-3,3,length)
    X = X[np.newaxis,:]
    param = [[1,2,3,4],[3.1,4.1,6.1,2.2],[1,1,1,1]]
    Y = np.zeros(length)[np.newaxis:]
    for p in range(len(param)):
        Y =Y + param[p][0]*np.sin(param[p][1]*np.pi*X) + param[p][2]*np.cos(param[p][3]*np.pi*X)
    #Y = 2*np.sin(2*np.pi*X) + 3*np.cos(4*np.pi*X)
    #Y_1 = 4*np.pi*np.cos(2*np.pi*X) - 12*
    #Y = 1*np.cos(2*np.pi*X)
    #X = norm(X)
    #Y = norm(Y)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(length):
        if i%4!=0 and i<length-50:
            x_train.append(X[0][i])
            y_train.append(Y[0][i])
        else:
            x_test.append(X[0][i])
            y_test.append(Y[0][i])
            
    x_train = np.array(x_train)[np.newaxis,:]
    y_train = np.array(y_train)[np.newaxis,:]
    x_test = np.array(x_test)[np.newaxis,:]
    y_test = np.array(y_test)[np.newaxis,:]
    
    plt.scatter(x_train,y_train,c='r',label = 'train')
    plt.ion()
    print('plot') 
    model = Model()
    model.add(DNN(100,input_dim=1,activation='tanh'))
    model.add(DNN(50,activation='tanh'))
    #model.add(DNN(128,activation='tanh'))
    model.add(DNN(1))
    model.compile(optimizer='adagrad',lr=0.005)
    loss_list = model.fit(x_train,y_train,epochs=90000)
    predict = model.predict(x_test)
    plt.plot(x_test.flatten(),predict.flatten(),'-',label = 'test')
    plt.legend()
    plt.savefig('F:\\天津大学课程\\图像合成\\作业\\fit.jpg', dpi=1200)    
    plt.show()
    
    
            
        