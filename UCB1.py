# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:55:41 2018

@author: prash
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
mat=scipy.io.loadmat('data.mat')
C=mat['C']
num_ads=np.shape(C)[1]
rounds=np.shape(C)[0]
Cmean=np.zeros((num_ads))

T=np.shape(C)[0]
reward=np.zeros((rounds))
reward_best=np.zeros((rounds))
regret=np.zeros((rounds))
UCB=np.zeros((num_ads))

data_mean=np.mean(C,axis=0)
adBest=np.argmax(data_mean)
reward[0:num_ads]=C[1,:]

num_selections=np.ones((num_ads))

t=num_ads

while(t<(rounds)):
    UCB=Cmean+np.sqrt(np.log(t)/num_selections)
#    if(t==1):
#        j=np.random.randint(ads)
#    else:
#        
    j=np.argmax(UCB)
    reward[t]=C[t,j]
    num_selections[j]=num_selections[j]+1
    Cmean[j]=Cmean[j]+(1/num_selections[j])*(reward[t]-Cmean[j])
    
    reward[t]=reward[t-1]+C[t,j]
    reward_best[t]=reward_best[t-1]+C[t,adBest]
    regret[t]=(reward_best[t]-reward[t])/(t)
    
    if  (t)%100==0:
        plt.plot(Cmean,label="empirical mean for chosen action")
        plt.ylabel("True Mean")
        plt.xlabel("Time")
        plt.legend()
        plt.show()
        plt.plot(num_selections,label="Number of selections")
        plt.ylabel("Number of times an arm is selected")
        plt.xlabel("Time")
        plt.legend()
        plt.show()

    t=t+1
    
plt.plot(reward,label="reward")
plt.plot(reward_best,label="Best reward")
plt.xlabel("Time")
plt.legend()
plt.show()
plt.plot(regret,label="Regret")
plt.xlabel("Time")
plt.legend()
plt.show()
