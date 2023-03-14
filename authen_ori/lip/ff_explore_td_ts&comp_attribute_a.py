import numpy as np
import math
np.random.seed(2018)
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib
import matplotlib.pyplot as plt
import dlib
from VideoPerson import *

from matplotlib import ticker

def get_3Metrics(data,label):
    ths=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]
   # ths=np.arange(0,0.6,0.01)
    measure_results=[]
    for th in ths:
        tp,fp,tn,fn=0,0,0,0
        for i in range(len(data)):
            if data[i][0]>=th and label[i]==1:
                tp+=1
            elif data[i][0]>=th and label[i]==-1:
                fp+=1
            elif data[i][0]<th and label[i]==1:
                fn+=1
            else:
                tn+=1
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        acc=(tp+tn)/(len(data))
        measure_results.append([precision,recall,acc])
    # plt.figure()
    # plt.plot(ths,[measure_results[i][0] for i in range(len(measure_results))],label='presion')
    # plt.plot(ths,[measure_results[i][1] for i in range(len(measure_results))],label='recall')
    # plt.plot(ths,[measure_results[i][2] for i in range(len(measure_results))],label='accuracy')
    # print(measure_results)
    # plt.show()
    return measure_results
def find_best_range(ths,metrics,bench=1):
    precision=[metrics[i][0] for i in range(len(metrics))]
    recall=[metrics[i][1] for i in range(len(metrics))]
    acc=[metrics[i][2] for i in range(len(metrics))]
    th_range=[]
    best_range=[]
    if bench==0:
        # take acc as the main: find the 0.95%best result
        acc_limits = 0.998
        recall_limits=0.999
        best_idx = np.argmax(acc)
        in_acc=[]
        for i in range(len(ths)):
            if recall[i] >= recall[best_idx] * recall_limits:
                th_range.append(ths[i])
                in_acc.append(acc[i])
        best_pre_idx = np.argmax(in_acc)
        for i in range(len(in_acc)):
            if in_acc[i]>=in_acc[best_pre_idx]* acc_limits:
                best_range.append(th_range[i])
    elif bench==1:
        # take 0.95% acc范围去寻找最好的precision
        acc_limits = 0.999
        recall_limits=0.998
        best_idx = np.argmax(acc)
        in_recall=[]
        for i in range(len(ths)):
            if acc[i] >= acc[best_idx] * acc_limits:
                th_range.append(ths[i])
                in_recall.append(recall[i])
        best_pre_idx = np.argmax(in_recall)
        for i in range(len(in_recall)):
            if in_recall[i]>=in_recall[best_pre_idx]* recall_limits:
                best_range.append(th_range[i])


    print(th_range)
    print(best_range)
    # plt.figure()
    # plt.plot(ths,precision,label='presion')
    # plt.plot(ths,recall,label='recall')
    # plt.plot(ths,acc,label='accuracy')
    # print(measure_results)
    # plt.show()
    return th_range,best_range
def processData(crf_distances,fake_distances,level='video'):
    crf_score=[]
    fake_score=[]
    if level=='video':
        for d in crf_distances:
            crf_score.append([np.mean(d)])

        for d in fake_distances:
            fake_score.append([np.mean(d)])
    elif level=='frame':
        for d in crf_distances:
            crf_score.extend(d)
        for d in fake_distances:
            fake_score.extend(d)
    data=crf_score+fake_score
    label=[-1 for d in crf_score]+[1 for d in fake_score]


    # print("Authenticating with machine learning technique----------------------------------------")
    # roc_auc,accuracy,recall=ml_analysis(data,label)
    # print("roc_auc:{}, accuracy:{}, recall:{}".format(roc_auc, accuracy, recall))
    return data,label
def getAveAuthenPerformance(crf_distances,fake_distances,repeat=1,level='video'):
    aucs,accs,recalls=[],[],[]
    for i in range(repeat):
        roc_auc, accuracy, recall = getAuthenPerformance(crf_distances, fake_distances, level)
        aucs.append(roc_auc)
        accs.append(accuracy)
        recalls.append(recall)
    return np.average(roc_auc),np.average(accs),np.average(recalls)
def clear_anomilies(distances,anomilies):
    new_distances=[]
    id=0
    for d in distances:
        if id in anomilies:
            id += 1
            # plt.figure()
            # plt.plot(d)
            # plt.show()
            continue
        new_distances.append(d)
        id+=1
    return new_distances
def collect_metrics_data(compression_level):
    ths = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]
    fake_anomilies = [322, 725]

    metrics_set=[]
    for i in range(len(ths)):
        th, crf_distances, fake_distances, saves = np.load(
            'data/faceforensics_comp' + str(compression_level) + '_threshold' + str(i) + '_distances_npy_cur_all_attribute_a.npy',
            allow_pickle=True)
        fake_distances = clear_anomilies(fake_distances, fake_anomilies)
        crf_distances, fake_distances=np.array(crf_distances), np.array(fake_distances)


        data, label = processData(crf_distances, fake_distances)#[np.random.choice(fake_distances.shape[0], crf_distances.shape[0], replace=False)])
        metrics = get_3Metrics(data, label)
        metrics_set.append(metrics)
    return metrics_set

#
# aucs_set=[]
# compression_levels=[0,1,2]
# save_set=[]
# metrics_set_levels=[]
# for compression_level in [0,1,2]:
#     print("Compression level: ",compression_level)
#     metrics_set=collect_metrics_data(compression_level)
#     metrics_set_levels.append(metrics_set)
# np.save('data/lip_3metrics_vs_comp&ts_attribute_a.npy',metrics_set_levels)
#find best range

metrics_set_levels=np.load('data/lip_3metrics_vs_comp&ts_attribute_a.npy',allow_pickle=True)
ths = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]


y_lims=[(0.1,.6),(.3,.5),(.4,.6)]
fig_all = plt.figure(figsize=(10,4.5))#figsize=(9, 5)figsize=(10,4.5)
ax_all = fig_all.add_subplot(111)
colors=['red','green','blue']
labels=['CRF=23','CRF=30','CRF=40']
lns=[]
for compression_level in [0, 1, 2]:

    for ts_i in range(len(ths)):
        best_range,best=find_best_range(ths,metrics_set_levels[compression_level][ts_i],1)
        print([d[2] for d in metrics_set_levels[compression_level][ts_i]])
        print("here",best_range,best,[np.min(best),np.max(best)])
        if compression_level==0:
            lns1 = ax_all.plot([ths[ts_i] for i in range(2)], [np.min(best),np.max(best)], linestyle='solid',marker='o',
                               color=colors[compression_level], linewidth=3, label=labels[compression_level],alpha=0.8)
        elif compression_level==1:
            lns2 = ax_all.plot([ths[ts_i] for i in range(2)], [np.min(best),np.max(best)], linestyle='solid',marker='v',
                               color=colors[compression_level],  linewidth=3,  label=labels[compression_level],alpha=0.8)
        else:
            lns3 = ax_all.plot([ths[ts_i] for i in range(2)], [np.min(best),np.max(best)], linestyle='solid',marker='s',
                               color=colors[compression_level],  linewidth=3, label=labels[compression_level],alpha=0.8)
       # ax_all.plot([ths[ts_i] for i in range(2)], [np.min(best),np.max(best)], marker='s', color=colors[compression_level],markersize=7)


    #fig.show()
ax_all.set_ylim(0.07,.14)

ax_all.grid()
ax_all.set_xlabel(r'$t_s$', fontsize=19)#fontsize=19
ax_all.set_ylabel(r'$t_d$', fontsize=19)
ax_all.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax_all.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
ax_all.tick_params(labelsize=17)#fontsize=16
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax_all.legend(lns, labs, loc="lower right", fontsize=17)

fig_all.savefig('figs/lip_td_range.pdf',bbox_inches='tight')
fig_all.show()

#
#
# for compression_level in [0, 1, 2]:
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for ts_i in range(len(ths)):
#
#         best_range,best=find_best_range(ths,metrics_set_levels[compression_level][ts_i],1)
#
#        # plt.plot([ths[ts_i] for i in range(len(best_range))],best_range,linestyle='solid',color='black')
#         ax.plot([ths[ts_i] for i in range(len(best))], best, linestyle='solid', color='black',linewidth=3)
#         ax.plot([ths[ts_i] for i in range(2)], [np.min(best),np.max(best)], marker='o', color='black')
#
#     ax.set_ylim(y_lims[compression_level])
#     #ax.set_xlim(0., .55)
#     # ax.legend(loc=0,fontsize=14)
#     ax.grid()
#     ax.set_xlabel(r'$t_s$', fontsize=14)
#     ax.set_ylabel(r'$t_d$', fontsize=14)
#
#    # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
#     ax.tick_params(labelsize=14)
#     fig.savefig('figs/lip_td_range_comp'+str(compression_level)+'.pdf')
#
