import math
import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np
import matplotlib.ticker as ticker
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

def getErrorRates(BER,n):
    error_bits_P=[]
    for i in range(n+1):
        error_bits_P.append(((1-BER)**(n-i))*(BER**i)*comb(n,i))
    #print(np.average(error_bits_P),np.std(error_bits_P))
    return error_bits_P
def correctCodeP(error_bits_P,t):
    correct_P=0
    for i in range(min(t+1,len(error_bits_P))):
        correct_P+=error_bits_P[i]
    return correct_P
def ErrorCodeP(error_bits_P,t):
    correct_P=0
    for i in range(t+1,len(error_bits_P)):
        correct_P+=i*error_bits_P[i]
    return correct_P/len(error_bits_P)
def test_BCH_n(BER):
    plt.figure()
    plt.subplot(3, 1, 1)
    error_bits_P = getErrorRates(BER, 127)  # 511---k=304; 255---k=139, 127-64
    plt.plot([i for i in list(range(len(error_bits_P)))], error_bits_P)
    plt.subplot(3, 1, 2)
    error_bits_P = getErrorRates(BER, 180)
    # plt.plot(list(range(len(error_bits_P))),error_bits_P)
    plt.plot([i for i in list(range(len(error_bits_P)))], error_bits_P)
    plt.subplot(3, 1, 3)
    error_bits_P = getErrorRates(BER, 72)
    # plt.plot(list(range(len(error_bits_P))),error_bits_P)
    plt.plot([i for i in list(range(len(error_bits_P)))], error_bits_P)
def make_test_R_0_545():
    BER = 0.025
    # n=255
    error_bits_P = getErrorRates(BER, 255)  # 511---k=304; 255---k=139, 127-64
    #
    t = 15  # n=255,k=139, k/n=0.545098
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot([i / 255 for i in list(range(len(error_bits_P)))], error_bits_P)

    print("The probability that BCH can correct the code is ", correctCodeP(error_bits_P, t))

    # same bit rate
    choices = [[21, 10, 5], [43, 20, 6]]  # k/n=0.545098
    plt.figure()
    plt.subplot(2, 1, 1)
    for choice in choices:
        n, t, s = choice
        SER = getSER(BER, s)
        error_bits_P = getErrorRates(SER, n)
        print("cfa", t, correctCodeP(error_bits_P, int(t / 2)) ** (int(255 / (n * s)) + 1))
        plt.plot([i for i in list(range(len(error_bits_P)))], error_bits_P,
                 label="RS(" + str(n) + "," + str(t) + "," + str(s) + ")")
    plt.legend()
    plt.show()
def make_test_R_0_45():
    BER = 0.04
    # n=255
    error_bits_P = getErrorRates(BER, 255)  # 511---k=304; 255---k=139, 127-64
    #
    t = 21  # n=255,k=115, k/n=0.45098
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot([i / 255 for i in list(range(len(error_bits_P)))], error_bits_P)

    print("The probability that BCH can correct the code is ", correctCodeP(error_bits_P, t))

    # same bit rate
    choices = [[21, 12, 5], [43, 23, 6]]  # k/n=0.545098
    plt.figure()
    plt.subplot(2, 1, 1)
    for choice in choices:
        n, t, s = choice
        SER = getSER(BER, s)
        error_bits_P = getErrorRates(SER, n)
        print("cfa", t, correctCodeP(error_bits_P, int(t / 2)) ** (int(255 / (n * s)) + 1))
        plt.plot([i for i in list(range(len(error_bits_P)))], error_bits_P,
                 label="RS(" + str(n) + "," + str(t) + "," + str(s) + ")")
    plt.legend()
    plt.show()
def make_test_R_BCH(BER,marker_this):
    n=255
    BCHs_P=[]
    BCHs_R=[]
    ts=[23,22,21,19,18,15,14,13,12]
    ks=[99,107,115,123,131,139,147,155,163]
    error_bits_P = getErrorRates(BER, n)  # 511---k=304; 255---k=139, 127-64
    print("BCH error bits:", error_bits_P)
    for i in range(len(ts)):
        BCHs_P.append(correctCodeP(error_bits_P, ts[i]))
        BCHs_R.append(ks[i]/n)
        #print(BCHs_P[-1],BCHs_R[-1])
    print(BCHs_P)
    plt.plot(BCHs_R,BCHs_P,linestyle='solid', marker=marker_this,label="BCH")
def make_test_R_RS(BER,marker_this):
    ts = [26,24,22,22,20,18,18,16]
    ts = [26,25, 24, 23, 22, 21, 20, 19, 18, 17, 16]
    # same bit rate
    n,s=43,6
    SER = getSER(BER, s)
    RSs_P,RSs_R=[],[]

    error_bits_P = getErrorRates(SER, n)

    print("ES:",SER)
    print("RS error symbol:",error_bits_P)

    for i in range(len(ts)):
        RSs_P.append(correctCodeP(error_bits_P, int(ts[i] / 2)) ** (int(255 / (n * s)) + 1))
        RSs_R.append((n-ts[i])/n)
    print(RSs_P)
    plt.plot(RSs_R,  RSs_P,linestyle='solid', marker=marker_this,label="RS")
def make_test_R(BER,marker1,marker2):

    n=255
    BCHs_P=[]
    BCHs_R=[]
    ts=[22,21,19,18,15,14,13,12]
    ks=[107,115,123,131,139,147,155,163]
    error_bits_P = getErrorRates(BER, n)  # 511---k=304; 255---k=139, 127-64
    for i in range(len(ts)):
        BCHs_P.append(correctCodeP(error_bits_P, ts[i]))
        BCHs_R.append(ks[i]/n)
        print(BCHs_P[-1],BCHs_R[-1])

    plt.plot(BCHs_R,BCHs_P,linestyle='solid', marker=marker1,label="BCH(BER="+str(BER)+")")

    ts = [26,24,22,22,20,18,18,16]

    # same bit rate
    n,s=43,6
    SER = getSER(BER, s)
    RSs_P,RSs_R=[],[]

    error_bits_P = getErrorRates(SER, n)
    for i in range(len(ts)):
        RSs_P.append(correctCodeP(error_bits_P, int(ts[i] / 2)) ** (int(255 / (n * s)) + 1))
        RSs_R.append((n-ts[i])/n)
    plt.plot(RSs_R,  RSs_P,linestyle='solid', marker=marker2,label="RS(BER="+str(BER)+")")
def make_test_P_BCH(BER,th):
    n=255
    ts=[12,13,14,15,18,19,21,22]
    ks=[163,155,147,139,131,123,115,107]
    error_bits_P = getErrorRates(BER, n)  # 511---k=304; 255---k=139, 127-64
    for i in range(len(ts)):
        if correctCodeP(error_bits_P, ts[i])>th:
            return ks[i]/n
    return False
def make_test_P_RS(BER,th):

    ts = range(10,43)
    n,s=43,6
    SER = getSER(BER, s)
    error_bits_P = getErrorRates(SER, n)
    for i in range(len(ts)):
        if correctCodeP(error_bits_P, int(ts[i] / 2)) ** (int(255 / (n * s)) + 1)>th:
            return (n-ts[i])/n
def make_test_P_BER():
    plt.figure(figsize=(10,5))
    BERs = np.arange(0.025,0.041,0.001)
    print(BERs)
    BCH_Rs=[]
    RS_Rs=[]
    th=0.99
    for BER in BERs:
        BCH_Rs.append(make_test_P_BCH(BER,th))
        RS_Rs.append(make_test_P_RS(BER, th))
    plt.plot(BERs,BCH_Rs,linestyle='solid', marker='o',label=r"BCH ($GF(2^8)$)")
    plt.plot(BERs, RS_Rs,linestyle='solid', marker='^',label=r"RS ($GF(2^6)$)")
    print(BCH_Rs[-1]*255)
    plt.xlabel("BER", fontsize=16)
    plt.ylabel("R", rotation=0, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig("BCH_RS_P_BER.pdf", bbox_inches='tight')
    plt.show()
def make_test_R_BER():
    BERs=[0.025,0.03,0.035,0.04]
    markers_BCH=['*','>','<','^','1']
    markers_RS = ['o', 'v', '8', 's', 'p']
    plt.figure(figsize=(10,5))

    for i in range(len(BERs)):
        BER=BERs[i]
        make_test_R_BCH(BER,markers_BCH[i])
    for i in range(len(BERs)):
        BER=BERs[i]
        make_test_R_RS(BER,markers_RS[i])
    plt.xlabel("R", fontsize=16)
    plt.ylabel("P", rotation=0, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig("BCH_RS_BER.pdf", bbox_inches='tight')
    plt.show()
def make_test_R_BER_simplify():
    BERs=[0.02,0.03,0.04]
    markers_BCH=['*','>','<','^','1']
    markers_RS = ['o', 'v', '8', 's', 'p']
    fig = plt.figure(figsize=(10,3))
    #plt.ylabel("P", fontsize=16)
    for i in range(len(BERs)):
        ax=plt.subplot(1,3, i+1)
        BER=BERs[i]
        make_test_R_BCH(BER,'*')
        make_test_R_RS(BER,'^')
        plt.xticks(fontsize=16)
        #plt.yticks([])
        plt.grid()
        plt.xlabel(r"R ($P_b=$"+str(BER)+")", fontsize=16)
        fff=plt.gca()
        if i==0:
            ax.set_ylabel("P", rotation=0, fontsize=16)

            plt.yticks(fontsize=16)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        plt.xlim((0.38,0.645))
        plt.ylim((0.37, 1.05))
        if i>0:
            ax.set_yticklabels([])

        plt.legend(fontsize=16)
        # if i>0:
        #     fff.axes.get_yaxis().set_visible(False)

    plt.yticks(fontsize=16)

    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.05, hspace=0)  # 调整子图间距
    # plt.xlabel("R", fontsize=16)
    # plt.ylabel("P", rotation=0, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig("BCH_RS_BER_subplot_array.pdf", bbox_inches='tight')
    plt.show()
print("caner", 1-correctCodeP(getErrorRates(getSER(0.1**4, 7),128),8))
print("BER out",ErrorCodeP(getErrorRates(getSER(0.1**3, 6),63),4))
print("caner", 1-correctCodeP(getErrorRates(0.1**4, 31),3))
make_test_R_BER_simplify()
make_test_P_BER()
make_test_R_BER()
#make_test_R_0_545()
#make_test_R_0_45()