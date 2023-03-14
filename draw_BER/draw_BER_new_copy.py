import math
import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np
def getSER(BER,symbol_size):
    return 1-(1-BER)**symbol_size
def getCER(SER,n,t):
    #n:message length, t:ECC length
    P_right=0
    for k in range(int(t/2)+1):

        P_right+=comb(n,k)*((1-SER)**(n-k))*(SER**k)
        #print(SER, P_right,comb(n, k), (1 - SER) ** (n - k), (SER ** k))
    return P_right

def getBER_P(n,t,s,BERs):
  #  BERs=[0.01,0.05,0.1,0.15,0.2,0.25]

    Ps=[]
    for BER in BERs:
        SER=getSER(BER,s)
        print("SER",BER,s,SER)
        Ps.append(getCER(SER,n,t))
    print(Ps)
    return Ps,BERs
def BCH(n):
    pos=math.log(n+1,2)-4
    k_set=[[11,7,5],[26,21,16,11,6],[57,51,45,39,36,30]]
    t_set=[[1,2,3],[1,2,3,5,7],[1,2,3,4,5,6]]

def drawBER(BER,n):
    error_bits_P=[]
    for i in range(30):
        error_bits_P.append(((1-BER)**(n-i))*(BER**i)*comb(n,i))
    print(np.average(error_bits_P),np.std(error_bits_P))
    return error_bits_P
BER=0.3
plt.figure()
plt.subplot(3,1,1)
error_bits_P=drawBER(BER,360)
plt.plot([i for i in list(range(len(error_bits_P)))],error_bits_P)
plt.subplot(3,1,2)
error_bits_P=drawBER(BER,180)
#plt.plot(list(range(len(error_bits_P))),error_bits_P)
plt.plot([i for i in list(range(len(error_bits_P)))],error_bits_P)
plt.subplot(3,1,3)
error_bits_P=drawBER(BER,72)
#plt.plot(list(range(len(error_bits_P))),error_bits_P)
plt.plot([i for i in list(range(len(error_bits_P)))],error_bits_P)
plt.show()
#same bit rate
#choices=[[5,4,3],[15,4,4],[31,8,5],[63,10,6],[127,12,7],[255,14,8]]
choices=[[32,14,5],[64,28,6],[128,56,7],[256,112,8],]
plt.figure()
plt.subplot(2,1,1)
for choice in choices:
    n,t,s=choice

  #  BERs = [0.18 ** 3, 0.19 ** 3, 0.2 ** 3, 0.21 ** 3]
    BERs = [0.006, 0.007, 0.008, 0.009, 0.01]
    BERs = [0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
    Ps,BERs=getBER_P(n,t,s,BERs)
    plt.plot(BERs,Ps,label="("+str(n)+","+str(t)+","+str(s)+")")
plt.subplot(2,1,2)
for choice in choices:
    n,t,s=choice
   # BERs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    BERs=[0.02,0.021,0.022,0.023,0.024, 0.025,0.026,0.027,0.028,0.029,0.03]
    BERs = [0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026]
    #BERs = [0.06, 0.061,0.062,0.063,0.064,0.065]
    Ps,BERs=getBER_P(n,t,s,BERs)
    plt.plot(BERs,Ps,label="("+str(n)+","+str(t)+","+str(s)+")")
plt.legend()
plt.show()

# choices=[[55,4,6],[55,6,6],[55,8,6],[55,10,6],[55,12,6],[55,14,6],[55,32,6],]
# plt.figure()
# for choice in choices:
#     n,t,s=choice
#     Ps,BERs=getBER_P(n,t,s)
#     plt.plot(BERs,Ps,label="("+str(n)+","+str(t)+","+str(s)+")")
# plt.legend()
# plt.show()