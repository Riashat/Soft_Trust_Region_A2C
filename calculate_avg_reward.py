
# coding: utf-8

# In[5]:


import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import glob
import pdb
import os
import pandas as pd
import matplotlib.pyplot as plt


# In[13]:


def read_file(infiles):
    datas=[]
    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    return datas


# In[14]:


def save_mean_rewards(num_processes=16):
    
    #all_data=pd.df
    all_data=[]
    data=[]
    indir='/usr/local/data/raihan/NIPS/Soft_Trust_Region_A2C/results_temp'
    for i in range(num_processes):
        infiles=glob.glob(os.path.join(indir, str(i)+'.monitor.csv'))
        data=read_file(infiles)
        data=df(data)
        all_data.append(data[2])

    mean_rewards=np.mean(all_data,axis=0)
    np.save('Mean_Rewards.npy',mean_rewards)


# In[15]:


save_mean_rewards()


# In[17]:


x=np.load('Mean_Rewards.npy')
plt.plot(x)
plt.xlabel('Timesteps')
plt.ylabel('Rewards')
plt.show()


# In[ ]:




