import math
import matplotlib.pyplot as plt
def get_lenghth(BER):
    return 2*BER/(1-2*BER)

n=[0,0.05, 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]
t=list(map(get_lenghth,n))
plt.figure(figsize=(10,4.5))
plt.plot(n,t,'orange')
plt.plot(n,t,'ro')
plt.xlabel("BER",fontsize=16)
plt.ylabel(r"$\frac{t}{n}$",rotation=0,fontsize=21)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
print(t[3])
#plt.legend(fontsize=14)
plt.text(0.11, 0.8, r'$(0.15, 0.43)$',fontsize=16)
plt.savefig("RS_BER.pdf",bbox_inches='tight')
plt.show()