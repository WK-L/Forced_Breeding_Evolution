# Forced Breeding Evolution

import os
import numpy as np
import matplotlib.pyplot as plt
from cec2017 import functions

class FBE:
    def __init__(self,fitFunc,dim=10,n=5,C=4,S=1):
        self.fitFunc = fitFunc
        self.dim = dim
        self.n = n*dim
        self.S = S # initial scale
        self.C = C # convergence time
        self.record_min = []
        self.record_avg = []
        self.y_best = None
        self.x = np.random.uniform(-100,100,(self.n,dim))
        self.y = self.fitFunc(self.x)
        self.update_best()

    def divide(self): # 隨機分兩群
        index = np.arange(self.n)
        np.random.shuffle(index)
        m = np.copy(index[:int(self.n/2)])
        f = np.copy(index[int(self.n/2):])
        return m, f

    def group(self,m,f): # 配對成兩兩一組
        y_f_sorted = np.argsort(self.y[f])
        pairs = np.zeros((self.n),dtype=np.int16)
        for i in y_f_sorted:
            # j = self.wheel(m,1)[0]
            j = self.tournament(m,1)[0]
            pairs[f[i]] = m[j]
            pairs[m[j]] = f[i]
            m = np.delete(m,j,axis=0)
        return pairs

    def crossover(self,pairs,round,iteration):
        x_temp = np.zeros_like(self.x)
        y_temp = np.zeros_like(self.y)
        S_success = []
        for i in range(self.n):
            rand = (-1)**np.random.randint(2,size=(2,self.dim))*np.random.normal(self.S, 0.1, (2,self.dim))*(1-np.exp((round-iteration)/iteration*self.C))
            x1 = np.copy(self.x[i])
            x1 += (self.x[pairs[i]]-self.x[i])*rand[0] + (self.x_best-self.x[i])*rand[1]
            for j in range(self.dim):
                if x1[j] > 100:
                    x1[j] = (self.x[i][j]+100)/2
                elif x1[j] < -100:
                    x1[j] = (self.x[i][j]-100)/2
            y_x1 = self.fitFunc([x1])[0]
            if self.y[i]>y_x1:
                x_temp[i] = np.copy(x1)
                y_temp[i] = y_x1
                S_success.extend(rand)
            elif np.random.rand()<np.exp((self.y[i]-y_x1)/self.y[i]*round):
                x_temp[i] = np.copy(x1)
                y_temp[i] = y_x1
            else:
                x_temp[i] = np.copy(self.x[i])
                y_temp[i] = self.y[i]
        self.x = np.copy(x_temp)
        self.y = np.copy(y_temp)
        self.update_scale(S_success)
    
    def update_best(self,round=0): # 更新最佳解
        self.record_min.append(self.y.min())
        self.record_avg.append(np.mean(self.y,axis=0))
        if not self.y_best or self.y_best>self.y.min():
            self.round_best = round
            self.y_best = self.y.min()
            self.x_best = np.copy(self.x[self.y.argmin()])

    def update_scale(self, S_success):
        if len(S_success) != 0:
            S_success = np.array(S_success)
            S_success = abs(S_success)
            self.S = sum(S_success)/len(S_success)

    def wheel(self,pool,n): # 輪盤法
        y_pool = 1/self.y[pool]
        s = y_pool.sum()
        weight = [y_pool[i]/s for i in range(y_pool.shape[0])]
        index = np.random.choice(pool.shape[0],n,p=weight,replace=False)
        return index

    def tournament(self,pool,n): # 競賽法
        sub_pool = np.random.choice(pool.shape[0],int(np.ceil(pool.shape[0]*0.5)),replace=False)
        y_sub_pool = self.y[pool[sub_pool]]
        i = np.argsort(y_sub_pool)[:n]
        return sub_pool[i]

    def run(self,iteration):
        for round in range(iteration):
            males, females = self.divide()
            pairs = self.group(males,females)
            self.crossover(pairs,round,iteration)
            self.update_best(round)
    
    def history_graph(self): # 繪製迭代歷史圖
        fig = plt.figure(figsize=(40,10))
        plt.title('FBE')
        plt.xlabel('iterate')
        plt.ylabel('fitness')
        plt.plot([i for i in range(200,len(self.record_avg))],self.record_avg[200:],color='b',label='average')
        plt.plot([i for i in range(200,len(self.record_min))],self.record_min[200:],color='r',label='minimum')
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__),'FBE_his.jpg'))

if __name__ == '__main__':
    fitFunc = functions.all_functions[6]
    fbe = FBE(fitFunc,dim=10)
    fbe.run(2000)
    fbe.history_graph()
    print(f'在第{fbe.round_best}回找到最佳解')
    print(f'最佳解: {fbe.x_best}')
    print(f'評估值: {fbe.y_best}')