import numpy as np
import scipy
import matplotlib.pylab as plt
import sklearn 
from sklearn.mixture       import GMM   
from matplotlib import pyplot as plt
import random

def mog (n, k, p1, p2 = 1.0, d=2):
  g = GMM(k)
  g.means_   = np.random.randn(k,d)
  g.means_[:,0] = p1;
  g.means_[:,1] = p1;
  g.covars_  = np.array ([np.array ([1.0, 1.0]), np.array ([1.0, 1.0])])
  g.weights_ = abs(np.random.rand(k,d))
  g.weights_ = g.weights_/sum(g.weights_)
  return g.sample(n)

RUNS = 100
data = []
numData = 1000
skew = 1.0
random.seed()
step = 0.1

data1 = np.append (mog(numData/2, 1, 1.0), (np.zeros(numData/2) + 1).reshape((numData/2, 1)), axis = 1)
data2 = np.append (mog(numData - numData/2, 1, -1.0), (np.zeros(numData - numData/2) - 1).reshape((numData - numData/2, 1)), axis = 1)
data = np.append (data1, data2, axis = 0)

#print data
print 'Data loaded..'
data = np.random.permutation(data)
print 'Permuting done..'
dim = 2

w = -1.0 + 2.0 * np.random.random(data.shape[1]) #np.zeros (8)
seen = {}

fig, ax = plt.subplots(1,2)#, figsize=(40,20))

ax[0].set_xlim ([-4, 4])
ax[0].set_ylim ([-4, 4])
ax[1].set_xlim ([-4, 4])
ax[1].set_ylim ([-4, 4])

x = np.arange(-3.0, 3.0, 0.1)
y = np.array ([(- w[2] - w[0]*xx)/w[1] for xx in x])
ax[0].plot (x, y, color = "b")
ax[1].plot (x, y, color = "b")
ax[0].scatter (data1[:, 0], data1[:, 1], color = "r")
ax[0].scatter (data2[:, 0], data2[:, 1], color = "g")

fig.set_figheight(10)
fig.set_figwidth(20)

plt.savefig('0')
plt.close()


for run in range(RUNS):    
  fig, ax = plt.subplots(1,2)
  ax[0].set_xlim ([-4, 4])
  ax[0].set_ylim ([-4, 4])
  ax[1].set_xlim ([-4, 4])
  ax[1].set_ylim ([-4, 4])

  score = [[np.dot (w[:2], data[i, :2]) + w[2], i] for i in range(numData)]
  #belief = [np.sign (score[i]) for i in range(numData)]
  score.sort(key= lambda x: np.abs(x[0]))
  '''
  print data
  print score
  raw_input()
  
  for i in range(numData):
  print np.exp(-skew*score[i][0])
  raw_input()
  '''
  #see label
  flag, i = 0, 0
  while (flag == 0 and i < numData): 
    if (data[score[i][1]][2] == 1.0):
      seen[score[i][1]] = (1, "r") 
      #print score [i][1]
      if(score[i][0] < 0.0):
        w = w + step * np.append (data[score[i][1]][:2], 1)
        flag = 1
    
    if (data[score[i][1]][2] == -1.0):
      #print 'okay'
      seen[score[i][1]] = (1, "g")
      if(score[i][0] > 0.0):
        w = w - step * np.append (data[score[i][1]][:2], 1)
        flag = 1
    i+=1
    
  x = np.arange(-3.0, 3.0, 0.1)
  y = np.array ([(- w[2] - w[0]*xx)/w[1] for xx in x])
  ax[0].plot (x, y, color = "b")
  ax[0].scatter (data1[:, 0], data1[:, 1], color = "r")
  ax[0].scatter (data2[:, 0], data2[:, 1], color = "g")

  for pt in seen:
    ax[1].scatter (data[pt][0], data[pt][1], color = seen[pt][1])
  ax[1].plot (x, y, color = "b")
  fig.set_figheight(10)
  fig.set_figwidth(20)

  plt.savefig(str(run+1))
  plt.close()
