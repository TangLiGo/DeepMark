import math
import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np
import matplotlib.ticker as ticker

markers_BCH = ['*', '>', '<', '^', '1']
markers_RS = ['o', 'v', '8', 's', 'p']
fig = plt.figure(figsize=(10, 3))

for i in range(len(BERs)):
    ax = plt.subplot(1, 3, i + 1)

    plt.xticks(fontsize=16)

    plt.grid()
    plt.xlabel(r"R ($P_b=$" + str(BER) + ")", fontsize=16)
    fff = plt.gca()
    if i == 0:
        ax.set_ylabel("P", rotation=0, fontsize=16)
        plt.yticks(fontsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))#调整x轴刻度间距
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.xlim((0.38, 0.645))#调整x轴数值范围
    plt.ylim((0.37, 1.05))#调整y轴数值范围
    if i > 0:
        ax.set_yticklabels([])

    plt.legend(fontsize=16)
    # if i>0:
    #     fff.axes.get_yaxis().set_visible(False)

plt.yticks(fontsize=16)

fig.tight_layout()  # 调整整体空白
plt.subplots_adjust(wspace=0.05, hspace=0)  # 调整子图间距

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.savefig("fig.pdf", bbox_inches='tight')
plt.show()