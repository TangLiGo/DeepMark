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
from base_utils_celebv2 import *
from matplotlib import ticker
def clear_saves(saves,compression_level):
    ori_pathes, fake_paths, CRF_paths,mul_people_video = loadFeatureFiles(compression_level, '_features_all.npy')
    saves_pure=[]
    j=0
    for i in range(len(ori_pathes)):
        ori_path=ori_pathes[i]
        CRF_path=CRF_paths[i]

        if ori_path in mul_people_video or CRF_path in mul_people_video:
            continue
      #  print(len(saves),i,j)
        saves_pure.append(saves[j])
        j+=2
        for fake_path in fake_paths[i]:
            if fake_path in mul_people_video:
                continue
            j+=1
    return saves_pure
def pickKeyFeature(data,threshold,feature_type='face'):
   # print("data",data[0][1][0][1])
   # selected_features=[]
    if feature_type=='face':
        feature_id=0
    elif feature_type=='lip':
        feature_id=2
    else:
        return
    start_id=0
    key_value =[]

    while key_value==[] and start_id<len(data):
        try:
            #print("hhh",data[start_id][1])
            key_value = data[start_id][1][0][feature_id]
        except:
            start_id+=1
   # print("refer start",start_id)
    if start_id==len(data):
        return [],None
    selected_features=[data[start_id]]

    for i in range(start_id+1,len(data)):
        try:
            if euclidean_distance([data[i][1][0][feature_id]], [key_value])[0]==0:
                print("here strange")
            if euclidean_distance([data[i][1][0][feature_id]], [key_value])[0] > threshold:
                #print("picked0,data[i][1][0][feature_id]",data[i][1][0][feature_id])
                selected_features.append(data[i])
                key_value = data[i][1][0][feature_id]
        except:
            continue
    return selected_features,len(selected_features)/len(data)
def getFeatureDistances(ref_data,cur_data,feature_type='face'):
    if feature_type=='face':
        feature_id=0
    elif feature_type=='lip':
        feature_id=2
    else:
        return
    distances=[]
    i,j=1,0
    while i<len(ref_data) and j<len(cur_data):
      #  print("i:{}, j:{}".format(ref_data[i - 1][0], cur_data[j][0]))
        try:
            if cur_data[j][0]<ref_data[i][0]:
                dis=euclidean_distance([ref_data[i-1][1][0][feature_id]], [cur_data[j][1][0][feature_id]])[0]
                distances.append(dis)
                j+=1
            else:
                i+=1
        except:
            try:
                test=ref_data[i-1][1][feature_id]
            except:
                i+=1
            try:
                test=cur_data[ j][1][feature_id]
            except:
                j+=1

    while i<len(ref_data):
        try:
            dis = euclidean_distance([ref_data[i - 1][1][0][feature_id]], [cur_data[-1][1][0][feature_id]])[0]
            distances.append(dis)
            i+=1
        except:
            i+=1
            continue
    while j<len(cur_data):
        try:
            dis = euclidean_distance([ref_data[- 1][1][0][feature_id]], [cur_data[j][1][0][feature_id]])[0]
            distances.append(dis)
            j+=1
        except:
            j+=1
            continue
    return distances
def collectDistances(selection_threshold_ref,selection_threshold_cur,compression_level):
    ori_pathes, fake_paths, CRF_paths,mul_people_video = loadFeatureFiles(compression_level, '_features_all.npy')
    CRF_distances=[]
    fake_distances=[]
    saves=[]

    for i in range(len(ori_pathes)):
        ori_path=ori_pathes[i]
        CRF_path=CRF_paths[i]

        if ori_path in mul_people_video or CRF_path in mul_people_video:
            continue
      #  print("start................", )
        feature_ref,save1=pickKeyFeature(np.load(ori_path,allow_pickle=True),selection_threshold_ref)

        feature_cur,save2=pickKeyFeature(np.load(CRF_path,allow_pickle=True),selection_threshold_cur)
     #   print("start CRF",ori_path)
        if save1 is None or save2 is None:
            continue
        distances=getFeatureDistances(feature_ref,feature_cur)

        CRF_distances.append(distances)
     #   print("CRF", distances)
        saves.append(save1)
      #  saves.append(save2)
        for fake_path in fake_paths[i]:
         #   print("here stack")
            if fake_path in mul_people_video:
                continue
          #  print("processing",fake_path)
            feature_cur,save3 = pickKeyFeature(np.load(fake_path, allow_pickle=True), selection_threshold_cur)

            if save3 is None:
                continue
            distances = getFeatureDistances(feature_ref, feature_cur)
            fake_distances.append(distances)
           # saves.append(save3)
           # print("fake", distances)

   # print(fake_distances)
    return CRF_distances,fake_distances,saves

def getAuthenPerformance(CRF_distances,fake_distances,level='video',fixed_th=0.3):
    CRF_score=[]
    fake_score=[]
    if level=='video':
        for d in CRF_distances:
            CRF_score.append([np.mean(d)])

        for d in fake_distances:
            fake_score.append([np.mean(d)])
    elif level=='frame':
        for d in CRF_distances:
            CRF_score.extend(d)
        for d in fake_distances:
            fake_score.extend(d)
    data=CRF_score+fake_score
    label=[-1 for d in CRF_score]+[1 for d in fake_score]
    print("Authenticating with fixed_threshold----------------------------------------")
    th,accuracy,recall=threshold_analysis(data,label)
    #print(data)
    print(compute_accuracy(label, [d[0] for d in data], fixed_th))
    print("best threshold:{}, accuracy:{}, recall:{}".format(th,accuracy,recall))


    print("Authenticating with machine learning technique----------------------------------------")
    roc_auc,accuracy,recall=ml_analysis(data,label)
    print("roc_auc:{}, accuracy:{}, recall:{}".format(roc_auc, accuracy, recall))
    return roc_auc,accuracy,recall
def getAveAuthenPerformance(CRF_distances,fake_distances,repeat=1,level='video'):
    aucs,accs,recalls=[],[],[]
    for i in range(repeat):
        roc_auc, accuracy, recall = getAuthenPerformance(CRF_distances, fake_distances, level)
        aucs.append(roc_auc)
        accs.append(accuracy)
        recalls.append(recall)
    return np.average(roc_auc),np.average(accs),np.average(recalls)
def draw_th_vs_auc(ths,aucs,saving,compression_level):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    lns1 = ax.plot(ths, aucs, 'ro', linestyle='solid', label='AUC')
    ax.set_ylim(0.98, 1.)
    # ax.legend(loc=0,fontsize=14)
    ax.grid()
    ax.set_xlabel(r'$t_s$', fontsize=14)
    ax.set_ylabel('AUC', fontsize=14)
   # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.tick_params(labelsize=14)
    # ax.set_yticks(fontsize=14)
    ax2 = ax.twinx()
    lns2 = ax2.plot(ths, [1 - d for d in saving], 'g^', linestyle='solid', label=r'$1-SR$')
    ax2.tick_params(labelsize=14)

    ax2.set_ylabel(r'$1-SR$', fontsize=14)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="lower right", fontsize=14)

    plt.savefig("figs/comp"+str(compression_level)+"_th_vs_auc.pdf")
  #  plt.show()
def clear_data(distances):
    out=[]
    for d in distances:
        if d==[]:
            continue
        out.append(d)
    return out
def explore_ths_comm_saving(compression_level):
    ths = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    thd=[0.3,0.35,0.45]
   # ths=[0.35]
    th_data = []
    i = 0
    fixed_th=0.4
    aucs=[]
    save_set=[]
    for th in range(len(ths)):
        if False:#os.path.exists('data/celebv2_comp'+str(compression_level)+'_threshold' + str(i) + '_distances.npy'):
            th, CRF_distances, fake_distances, saves = np.load('data/celebv2_comp'+str(compression_level)+'_threshold' + str(i) + '_distances.npy',
                                                               allow_pickle=True)
        else:
            selection_threshold_ref, selection_threshold_cur = th, th
            CRF_distances, fake_distances, saves = collectDistances(selection_threshold_ref, selection_threshold_cur,compression_level)
         #   np.save('data/celebv2_comp'+str(compression_level)+'_threshold' + str(i) + '_distances.npy', [th, CRF_distances, fake_distances, saves])
#        saves = clear_saves(saves, compression_level)
        CRF_distances,fake_distances=np.array(CRF_distances),np.array(fake_distances)
    #    show_distribution(CRF_distances, fake_distances,'figs/celebv2_comp'+str(compression_level)+'th'+str(i)+'.pdf')

        print("t_s=",th)
        fixed_th=thd[compression_level]
        CRF_distances, fake_distances=np.array(CRF_distances), np.array(fake_distances)
        #为了平均正负样本的数量
        # if compression_level==2:
        #     fk_collect = np.random.choice(fake_distances.shape[0], CRF_distances.shape[0], replace=False)
        #     np.save('fake_collect.npy',fk_collect)
        # else:
        #     fk_collect=np.load('fake_collect.npy',allow_pickle=True)
       # print("fk",fk_collect)
        fk_collect = np.load('fake_collect.npy', allow_pickle=True)
        roc_auc, accuracy, recall = getAuthenPerformance(CRF_distances, fake_distances[fk_collect])#fake_distances[np.random.choice(fake_distances.shape[0], CRF_distances.shape[0], replace=False)],fixed_th=fixed_th)
       # roc_auc, accuracy, recall = getAuthenPerformance(CRF_distances, fake_distances)
       #  show_distribution(CRF_distances,
       #            fake_distances[fk_collect],
       #            'figs/comp' + str(compression_level) + '_face_descriptor_dist_ts' + str(i) + '_balance.pdf')
        th_data.append([CRF_distances, fake_distances, saves, roc_auc, accuracy, recall])
        aucs.append(roc_auc)
        i += 1
        save_set.append(np.average(saves))
   # draw_th_vs_auc(ths,aucs,save_set,compression_level)
    print("Accuracy:",aucs)
    print("saving",save_set)
   # np.save('data/celebv2_'+str(compression_level)+'threshold_vs_comp.npy', th_data)
   # np.save('data/celebv2_'+str(compression_level)+'threshold_vs_comp.npy', [th_data,[ths,aucs,save_set]])
    return ths,aucs,save_set,compression_level
def explore_compression(compression_level):
    ths = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
   # ths=[0.35]
    th_data = []
    iset = [0,7]

    aucs=[]
    save_set=[]
    thd = [0.3, 0.35, 0.45]
    fixed_th = thd[compression_level]
    for i in iset:

        th, CRF_distances, fake_distances, saves = np.load('data/celebv2_comp'+str(compression_level)+'_threshold' + str(i) + '_distances.npy',
                                                               allow_pickle=True)
        CRF_distances,fake_distances=np.array(CRF_distances),np.array(fake_distances)
        show_distribution(CRF_distances, fake_distances,'figs/celebv2_comp'+str(compression_level)+'th'+str(i)+'.pdf')
        #为了平均正负样本的数量
        roc_auc, accuracy, recall = getAuthenPerformance(CRF_distances, fake_distances[np.random.choice(fake_distances.shape[0], CRF_distances.shape[0], replace=False)],fixed_th=fixed_th)
       # roc_auc, accuracy, recall = getAuthenPerformance(CRF_distances, fake_distances)
        th_data.append([CRF_distances, fake_distances, saves, roc_auc, accuracy, recall])
        aucs.append(roc_auc)
        i += 1

def rename_file(compression_level):
    ths = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
   # ths=[0.35]
    th_data = []
    i = 0
    fixed_th=0.4
    aucs=[]
    save_set=[]
    for th in ths:
        old_name='data/celebv2_comp'+str(compression_level)+'_threshold' + str(i) + '_distances.npy'
        new_name='data/celebv2_comp'+str(compression_level)+'_threshold' + str(i) + '_distances_'+'false_refpoint.npy'
        os.rename(old_name,new_name)
        i+=1


def show_distribution(CRF_distances, fake_distances,name,level='video'):
    crf_score=[]
    fake_score=[]

    id=0
    if level=='video':
        for d in CRF_distances:
            crf_score.append([np.mean(d),np.std(d)])
        for d in fake_distances:
            fake_score.append([np.mean(d),np.std(d)])
            id+=1
    elif level=='frame':
        for d in CRF_distances:
            crf_score.extend(d)
        for d in fake_distances:
            fake_score.extend(d)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.scatter([d[0] for d in crf_score],[d[1] for d in crf_score],label='Authentic',alpha=0.5)
    plt.scatter([d[0] for d in fake_score], [d[1] for d in fake_score], label='Face-Swapping',alpha=0.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.xticks(fontsize=21)
    plt.xlabel(r'$\mu$',fontsize=21)
    plt.ylabel(r'$\sigma$',fontsize=21)
    plt.yticks(fontsize=21)
    plt.legend(fontsize=21)
    plt.savefig(name,bbox_inches='tight')
    plt.show()

def draw_th_vs_auc_levels(ths,aucs_set,saving):
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    lns1 = ax.plot(ths, aucs_set[0], color='red', marker='o', linestyle='solid', label='AUC-CRF23')
    lns2 = ax.plot(ths, aucs_set[1], color='green', marker='v', linestyle='solid', label='AUC-CRF30')
    lns3 = ax.plot(ths, aucs_set[2], color='blue', marker='s', linestyle='solid', label='AUC-CRF40')
    ax.set_ylim(0.94, 1.0)

    # ax.legend(loc=0,fontsize=14)
    ax.grid()
    ax.set_xlabel(r'$t_s$', fontsize=19)
    ax.set_ylabel('AUC', fontsize=16)
  #  ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.tick_params(labelsize=16)
   # ax.set_yticks(fontsize=16)
    ax2 = ax.twinx()
    # print(saving)
    lns4 = ax2.plot(ths, [1 - d for d in saving], color='orange', marker='^', linestyle='solid', label=r'$1-SR$')
    ax2.tick_params(labelsize=16)
  #  ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax2.set_ylabel(r'$1-SR$', fontsize=16)
  #  ax2.set_yscale('log')
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="lower right", fontsize=16)
    plt.savefig("figs/face_comps_th_vs_auc.pdf",bbox_inches='tight')
    plt.show()


aucs_set=[]
compression_levels=[0,1,2]
save_set=[]
for compression_level in [0,1,2]:
    print("Compression level: ",compression_level)

    ths,aucs,saves,compression_level=explore_ths_comm_saving(compression_level)
#
#     dd=np.load('data/celebv2_'+str(compression_level)+'threshold_vs_comp.npy', allow_pickle=True)
#     set_info=dd[1]
#     ths, aucs, saves=set_info
    aucs_set.append(aucs)
    save_set.append(saves)
# #     draw_th_vs_auc(ths, aucs, save_set[0], compression_level)
#
# #np.save("data/comp_th_relationship.npy",[ths,aucs_set,save_set[0]])
#ths,aucs_set,save_set=np.load("data/comp_th_relationship.npy",allow_pickle=True)
print(ths)
print(aucs_set)
draw_th_vs_auc_levels(ths,aucs_set,save_set[1])
