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
from base_utils import *
from matplotlib import ticker
def get_person_data(video_data):
    persons=FaceTracker()
    person_num=0
    face_id=0
    for i in range(len(video_data)):
       # print(video_data[i][1])

        for j in range(len(video_data[i][1])):
            face_info=FaceFeature(video_data[i][1][j][0], video_data[i][1][j][2], video_data[i][1][j][1], video_data[i][0])
            match_result = persons._match([face_info], video_data[i][0], [])
            if len(match_result) == 0:
                continue
        persons._check_state(video_data[i][0])
    return persons
def get_mouth_area(lip_landmarks):
    left,right,top,bottom=lip_landmarks[0][0],lip_landmarks[0][0],lip_landmarks[0][1],lip_landmarks[0][1],
    for landmark in lip_landmarks:
        if landmark[0]<left:
            left=landmark

def get_person_data2(video_data):
    persons=FaceTracker()
    person_num=0
    face_id=0
    for i in range(len(video_data)):
       # print(len(video_data[i]),(video_data[i]))
        if video_data[i] is None:
            continue
        face_info = FaceFeature([0,1,2,3,4], video_data[i][1], [100, 50, 150, 100], i)
        match_result = persons._match([face_info],i, [])

    return persons
def get_lip_attributes(lip_landmarks, lip_size):

    d1,d2,d3,d4=[],[],[],[]
    # print("lip_landmarks",lip_landmarks)
    # print("lip_landmarks22",lip_landmarks[6][0])
    # print("lip_size",lip_size)
    d1.append((lip_landmarks[6][0]-lip_landmarks[0][0])/lip_size[0])#width
    d1.append((lip_landmarks[9][1] - lip_landmarks[3][1])/lip_size[1])#height
    for i in range(1,6):
        d2.append((lip_landmarks[12-i][1] - lip_landmarks[i][1])/lip_size[1])
    center_point=[((lip_landmarks[6][0]+lip_landmarks[0][0])/2),(lip_landmarks[9][1] + lip_landmarks[3][1])/2]

    for i in range(12):
        d3.append(euclidean_distance([[lip_landmarks[i][0]/lip_size[0],lip_landmarks[i][1]/lip_size[1]]],[[center_point[0]/lip_size[0],center_point[1]/lip_size[1]]])[0,0])
    for i in range(12,20):
        d4.append(euclidean_distance([[lip_landmarks[i][0] / lip_size[0], lip_landmarks[i][1] / lip_size[1]]],
                                     [[center_point[0] / lip_size[0], center_point[1] / lip_size[1]]])[0, 0])
   # print("d1,d2,d3,d4",d1,d2,d3,d4)
    return d1,d2,d3,d4
def pickKeyFeature(data,lip_size,attr_threshold, attr_i,feature_type='lip'):

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
          #  print("hhh",data[start_id][1])
            key_value = data[start_id][1][0][feature_id]
        except:
            start_id+=1
   # print("refer start",start_id)
    if start_id==len(data):
        return [],None
    selected_features=[data[start_id]]

    for i in range(start_id+1,len(data)):
        # lip_attributes1 = get_lip_attributes(data[i][1][0][feature_id], lip_size)
        # lip_attributes2 = get_lip_attributes(key_value, lip_size)
        # if euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0] > attr_threshold:
        #     #print("lip_attributes1[attr_i]",lip_attributes1[attr_i])
        #     selected_features.append(data[i])
        # print("data[i][1][0][feature_id]",get_lip_attributes(data[i][1][0][feature_id], lip_size)[attr_i])
        try:
            lip_attributes1 = get_lip_attributes(data[i][1][0][feature_id], lip_size)
            lip_attributes2 = get_lip_attributes(key_value, lip_size)
            if euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0]==0:
                print("here strange1",key_value)
                print("here strange2", key_value)
            if euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0] > attr_threshold:
                # print("lip_attributes1[attr_i]",lip_attributes1[attr_i])
                selected_features.append(data[i])
                key_value=data[i][1][0][feature_id]
        except:
            continue
  #  print(selected_features)
  #    print("cur", len(selected_features), len(data))
    return selected_features,len(selected_features)/len(data)
def get_lip_size(descriptors):
    w,h=0,0
    for i in range(len(descriptors)):
      #  print(descriptors[i])
        if len(descriptors[i][1])==0:
            continue
        lip_landmarks=descriptors[i][1][0][2]
        cur_w=(lip_landmarks[6][0] - lip_landmarks[0][0])  # width
        if cur_w>w:
            w=cur_w
        cur_h=((lip_landmarks[9][1] - lip_landmarks[3][1]))  # height
        if cur_h>h:
            h=cur_h
    return [w,h]


def getFeatureDistances(attr_i,ref_data, cur_data, lip_size,len_prob=1):
    #ref_data waiting for authentication
    #cur_data stored info
    #len_prob=len(ref_data.landmarks)/len(cur_data.landmarks)---ori/cur
    lip_distance = []

    i = 0
    j = 0
    while i < len(ref_data.features) and j < len(cur_data.features) - 1:
      #  print("comming compare success")
      #  print(ref_data.features[i].lip_descriptor,cur_data.features[j + 1].lip_descriptor)
      #  print(lip_size)
        if ref_data.features[i].frame_pos < int(len_prob*(cur_data.features[j + 1].frame_pos)):
            lip_attributes1=get_lip_attributes(ref_data.features[i].lip_descriptor, lip_size)
            lip_attributes2=get_lip_attributes(cur_data.features[j].lip_descriptor, lip_size)
            dis = euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0,0]
            lip_distance.append(dis)
            i += 1
        else:
            j += 1


    while i<len(ref_data.features):
        try:
            lip_attributes1=get_lip_attributes(ref_data.features[i].lip_descriptor, lip_size)
            lip_attributes2=get_lip_attributes(cur_data.features[-1].lip_descriptor, lip_size)
            dis = euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0,0]
            lip_distance.append(dis)
            i+=1
        except:
            i+=1
            continue
    while j<len(cur_data.features):
        try:
            lip_attributes1=get_lip_attributes(ref_data.features[-1].lip_descriptor, lip_size)
            lip_attributes2=get_lip_attributes(cur_data.features[j].lip_descriptor, lip_size)
            dis = euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0,0]
            j+=1
        except:
            j+=1
            continue
   # print("lip_distance",lip_distance)
    return lip_distance
def getFeatureDistances(ref_data,cur_data,attr_i,lip_size,feature_type='lip'):
    if feature_type=='face':
        feature_id=0
    elif feature_type=='lip':
        feature_id=2
    else:
        return
    distances=[]
    i,j=1,0
 #   print("i:{}, j:{}".format(len(ref_data), len(cur_data)))
    while i<len(ref_data) and j<len(cur_data):

        try:
            if cur_data[j][0] < ref_data[i][0]:
                # Note：！！！！！！！！！！！！！！！！！！
                lip_attributes1 = get_lip_attributes(ref_data[i-1][1][0][feature_id], lip_size)
                lip_attributes2 = get_lip_attributes(cur_data[j][1][0][feature_id], lip_size)
               # print("last_i:{}, j:{}, next_i:{}".format( ref_data[i-1][0], cur_data[j][0],ref_data[i][0]))
                dis = euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0]
               # print("dis:",dis)
                distances.append(dis)

                j += 1
            else:
                i += 1

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
            lip_attributes1 = get_lip_attributes(ref_data[i - 1][1][0][feature_id], lip_size)
            lip_attributes2 = get_lip_attributes(cur_data[-1][1][0][feature_id], lip_size)
            # print("last_i:{}, j:{}, next_i:{}".format( ref_data[i-1][0], cur_data[j][0],ref_data[i][0]))
            dis = euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0]
            distances.append(dis)
          #  print("last_i:{}, j:{}".format( ref_data[i-1][0], cur_data[j][0]))
            i+=1
        except:
            i+=1
            continue
    while j<len(cur_data):
        try:
            lip_attributes1 = get_lip_attributes(ref_data[- 1][1][0][feature_id], lip_size)
            lip_attributes2 = get_lip_attributes(cur_data[j][1][0][feature_id], lip_size)
            # print("last_i:{}, j:{}, next_i:{}".format( ref_data[i-1][0], cur_data[j][0],ref_data[i][0]))
            dis = euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0]
            distances.append(dis)
       #     print("last_i2:{}, j:{}".format( ref_data[i-1][0], cur_data[j][0]))
            j+=1
        except:
            j+=1
            continue
    return distances
def getFeatureDistancescopu(ref_data,cur_data,attr_i,lip_size,feature_type='lip'):
    if feature_type=='face':
        feature_id=0
    elif feature_type=='lip':
        feature_id=2
    else:
        return
    distances=[]
    i,j=1,0
 #   print("i:{}, j:{}".format(len(ref_data), len(cur_data)))
    while i<len(ref_data) and j<len(cur_data):

        try:
            if j < i:
                # Note：！！！！！！！！！！！！！！！！！！
                lip_attributes1 = get_lip_attributes(ref_data[i-1][1][0][feature_id], lip_size)
                lip_attributes2 = get_lip_attributes(cur_data[j][1][0][feature_id], lip_size)
                print("i:{}, j:{}".format(i-1, j))
                dis = euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0]
               # print("dis:",dis)
                distances.append(dis)

                j += 1
            else:
                i += 1

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
            print("here1 i:{}, j:{}".format(i-1, j))
            i+=1
        except:
            i+=1
            continue
    while j<len(cur_data):
        try:
            dis = euclidean_distance([ref_data[- 1][1][0][feature_id]], [cur_data[j][1][0][feature_id]])[0]
            distances.append(dis)
            print("here2 i:{}, j:{}".format(i-1, j))
            j+=1
        except:
            j+=1
            continue
    return distances

def record_mul_people(compression_level):
    ori_pathes, fake_paths, crf_paths = traverse_forensics_videos2(compression_level)
    mul_videos=[]
    for paths in [ori_pathes, crf_paths]:
        for path in paths:
            if check_multi_people(np.load(path, allow_pickle=True)):
                mul_videos.append(path)
    return mul_videos
def collectDistances(attr_i,selection_threshold_ref,selection_threshold_cur,compression_level):
    #ori_pathes, fake_paths, crf_paths,mul_people_video = loadFeatureFiles(compression_level, '_features_all_attribute_a.npy')
    ori_paths, fake_paths, crf_paths = traverse_forensics_videos3(compression_level)
   # print(ori_paths)
    mul_people_video = np.load("data/faceforensics_multi_person_video_face2face.npy", allow_pickle=True)
    mul_people_video_id = ['011', '021', '027', '033', '034', '038', '048', '060', '091', '105', '148', '156', '169',
                           '170', '175', '178', '186', '187', '188', '190', '210', '212', '217', '219', '220', '240',
                           '253', '257', '263', '274',
                           '305', '325', '332', '339', '344', '345', '346', '356', '357', '359', '370', '404', '412',
                           '425', '447', '448', '493', '496', '509', '522', '562', '566', '574', '618', '629',
                           '636', '659', '673', '692', '706', '720', '724', '732', '736', '738', '756', '759', '760',
                           '764', '765', '776', '787', '794', '799', '805', '812', '817', '824', '829', '869', '873',
                           '880', '897', '908', '930', '950', '994']

    crf_distances, fake_distances=[],[]

    _, lip_sizes = np.load('data/forensics_lip_size.npy', allow_pickle=True)

    saves=[]
    for i in range(len(ori_paths)):
        if ori_paths[i] in mul_people_video:
            continue
        if crf_paths[i] in mul_people_video:
            continue
       # print(ori_paths[i])
        lip_size = get_lip_size(np.load(ori_paths[i],allow_pickle=True))

        feature_ref,save1 = pickKeyFeature(np.load(ori_paths[i],allow_pickle=True),lip_size,selection_threshold_ref,attr_i)

        feature_cur= np.load(crf_paths[i],allow_pickle=True)#pickKeyFeature(np.load(crf_paths[i],allow_pickle=True),lip_size,selection_threshold_ref,attr_i)
      #  print(ori_paths[i])
      #  print("---------------crf---------------")
       # print(crf_paths[i])
        distances = getFeatureDistances(feature_ref,feature_cur,attr_i,lip_sizes[i])
        crf_distances.append(distances)
     #   print("crf_distances",distances)
        saves.append(save1)
        if fake_paths[i] in mul_people_video:
            continue
       # print("pickKeyFeature data", np.load(fake_paths[i], allow_pickle=True))
        feature_cur =np.load(fake_paths[i],allow_pickle=True)# pickKeyFeature(np.load(fake_paths[i],allow_pickle=True),lip_size,selection_threshold_ref,attr_i)
      #  print("---------------df---------------")
       # print(fake_paths[i])
        distances = getFeatureDistances(feature_ref,feature_cur,attr_i,lip_sizes[i])
        fake_distances.append(distances)
     #   print("fake_distances", distances)
   # ref_data, cur_data, lip_size, len_prob = 1
    return crf_distances,fake_distances,saves
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

def getAuthenPerformance(crf_distances,fake_distances,level='video'):
    crf_score=[]
    fake_score=[]

    if level=='video':
        id=0
        for d in crf_distances:
            crf_score.append([np.mean(d)])
            # if crf_score[-1][0]>0.14:
            #     print("id",id)
            #     plt.figure()
            #     plt.plot(d)
            #     plt.show()
            # id+=1
        for d in fake_distances:
            fake_score.append([np.mean(d)])


    elif level=='frame':
        for d in crf_distances:
            crf_score.extend(d)
        for d in fake_distances:
            fake_score.extend(d)


    data=crf_score+fake_score
    label=[-1 for d in crf_score]+[1 for d in fake_score]
    print("Authenticating with fixed_threshold----------------------------------------")
    th,accuracy,recall=threshold_analysis(data,label)
    print("best threshold:{}, accuracy:{}, recall:{}".format(th,accuracy,recall))


    print("Authenticating with machine learning technique----------------------------------------")
    roc_auc,accuracy,recall=ml_analysis(data,label)
    print("roc_auc:{}, accuracy:{}, recall:{}".format(roc_auc, accuracy, recall))
    return roc_auc,accuracy,recall
def getAveAuthenPerformance(crf_distances,fake_distances,repeat=1,level='video'):
    aucs,accs,recalls=[],[],[]
    for i in range(repeat):
        roc_auc, accuracy, recall = getAuthenPerformance(crf_distances, fake_distances, level)
        aucs.append(roc_auc)
        accs.append(accuracy)
        recalls.append(recall)
    return np.average(roc_auc),np.average(accs),np.average(recalls)
def draw_th_vs_auc(ths,aucs,saving,compression_level):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    lns1 = ax.plot(ths, aucs, 'ro', linestyle='solid', label='AUC')
    ax.set_ylim(0.994, 1.)
    # ax.legend(loc=0,fontsize=14)
    ax.grid()
    ax.set_xlabel(r'$t_s$', fontsize=14)
    ax.set_ylabel('AUC', fontsize=14)
   # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.tick_params(labelsize=14)
    # ax.set_yticks(fontsize=14)
    ax2 = ax.twinx()
    #print("saving",saving)
    lns2 = ax2.plot(ths, [1 - d for d in saving], 'g^', linestyle='solid', label=r'$1-SR$')
    ax2.tick_params(labelsize=14)

    ax2.set_ylabel(r'$1-SR$', fontsize=14)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="lower right", fontsize=14)

    plt.savefig("figs/comp"+str(compression_level)+"_th_vs_auc.pdf")
    plt.show()
def gen_th_data(compression_level):
    ths = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]
   # ths=[0.08]
    th_data = []
    aucs=[]
    save_set=[]
    fake_anomilies = [322, 725]
   # fake_anomilies = [110,212,322,725]
    for i in [0]:#range(len(ths)):
        th=ths[i]
        if os.path.exists('data/faceforensics_comp'+str(compression_level)+'_threshold' + str(i) + '_distances_npy_cur_all_attribute_a.npy'):
            th, crf_distances, fake_distances, saves = np.load('data/faceforensics_comp'+str(compression_level)+'_threshold' + str(i) + '_distances_npy_cur_all_attribute_a.npy',
                                                               allow_pickle=True)
        else:
            selection_threshold_ref, selection_threshold_cur = th, th
            crf_distances, fake_distances, saves = collectDistances(0,selection_threshold_ref, selection_threshold_cur,compression_level)
           # np.save('data/faceforensics_comp'+str(compression_level)+'_threshold' + str(i) + '_distances_npy_cur_all_attribute_a.npy', [th, crf_distances, fake_distances, saves])
      #  saves = clear_saves(saves, compression_level)
        fake_distances=clear_anomilies(fake_distances,fake_anomilies)
        show_distribution(crf_distances, fake_distances,'figs/comp'+str(compression_level)+'_lip_descriptor_dist_ts'+str(i)+'.pdf')
        print("t_s=",th)
        # print(saves)
        # plt.figure()
        # plt.hist([1-s for s in saves])
        # plt.show()
       # print(crf_distances, fake_distances)
        crf_distances, fake_distances=np.array(crf_distances), np.array(fake_distances)
      #  roc_auc, accuracy, recall =   getAuthenPerformance(crf_distances, fake_distances[np.random.choice(fake_distances.shape[0], crf_distances.shape[0], replace=False)])
       # print("crf_distances",crf_distances)
       # print("fake_distances",fake_distances)
        roc_auc, accuracy, recall = getAuthenPerformance(crf_distances, fake_distances)
        th_data.append([crf_distances, fake_distances, saves, roc_auc, accuracy, recall])
        aucs.append(roc_auc)
        i += 1
        save_set.append(np.average(saves))
   # draw_th_vs_auc(ths,aucs,save_set,compression_level)
    print("Accuracy:",aucs)
    print("saving",save_set)

   # np.save('data/faceforensics_'+str(compression_level)+'threshold_vs_comp_attribute_a.npy', [th_data,[ths,aucs,save_set]])
    return ths,aucs,save_set,compression_level
def show_distribution(crf_distances, fake_distances,name,level='video'):
    crf_score=[]
    fake_score=[]

    id=0
    if level=='video':
        for d in crf_distances:
            crf_score.append([np.mean(d),np.std(d)])
        for d in fake_distances:
            fake_score.append([np.mean(d),np.std(d)])
            id+=1
    elif level=='frame':
        for d in crf_distances:
            crf_score.extend(d)
        for d in fake_distances:
            fake_score.extend(d)
    # plt.figure()
    # plt.scatter([d[0] for d in crf_score],[d[1] for d in crf_score],label='Authentic',alpha=0.5)
    # plt.scatter([d[0] for d in fake_score], [d[1] for d in fake_score], label='Lip-Reenactment',alpha=0.5)
    #
    # plt.xticks(fontsize=21)
    # plt.xlabel(r'$\mu$',fontsize=21)
    # plt.ylabel(r'$\sigma$',fontsize=21)
    # plt.yticks(fontsize=21)
    # plt.legend(fontsize=21)
    # plt.savefig(name,bbox_inches='tight')
    print("name",name)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.scatter([d[0] for d in crf_score],[d[1] for d in crf_score],label='Authentic',alpha=0.5)
    plt.scatter([d[0] for d in fake_score], [d[1] for d in fake_score], label='Lip-Reenactment',alpha=0.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.12))
   # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.15))
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
    lns1 = ax.plot(ths[:16], aucs_set[0][:16], color='red',marker='o', linestyle='solid', label='AUC-CRF23')
    lns2 = ax.plot(ths[:16], aucs_set[1][:16], color='green',marker='v', linestyle='solid', label='AUC-CRF30')
    lns3 = ax.plot(ths[:16], aucs_set[2][:16], color='blue',marker='s', linestyle='solid', label='AUC-CRF40')
    ax.set_ylim(0.96, 1.)
    # ax.legend(loc=0,fontsize=14)
    ax.grid()
    ax.set_xlabel(r'$t_s$', fontsize=19)
    ax.set_ylabel('AUC', fontsize=16)
    ax.tick_params(labelsize=16)
    ax2 = ax.twinx()
    lns4 = ax2.plot(ths[:16], [1 - d for d in saving][:16], color='orange',marker='^', linestyle='solid', label=r'$1-SR$')
    ax2.tick_params(labelsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax2.set_ylabel(r'$1-SR$', fontsize=16)
    lns = lns1 + lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="lower right", fontsize=16)
    print(aucs_set[0][10], aucs_set[1][10], aucs_set[2][10])
    plt.savefig("figs/lip_comps_th_vs_auc.pdf",bbox_inches='tight')
    plt.show()
aucs_set=[]
compression_levels=[0,1,2]
save_set=[]
for compression_level in [0,1,2]:
    print("Compression level: ",compression_level)

    ths,aucs,saves,compression_level=gen_th_data(compression_level)

    th_data,set_info=np.load('data/faceforensics_' + str(compression_level) + 'threshold_vs_comp_attribute_a.npy', allow_pickle=True)
    ths, aucs, saves=set_info
    aucs_set.append(aucs)
    save_set.append(saves)
    #draw_th_vs_auc(ths, aucs, save_set[0], compression_level)
print(aucs_set[0][10],aucs_set[0])
#print("save_set[0]",save_set[0])

save_set[0][0]=1
print("save_set[0]",save_set[0])
draw_th_vs_auc_levels(ths,aucs_set,save_set[0])
#
# repeat_times=1
# aucs_set_ave,save_set_ave=[],[]
# compression_levels=[0,1,2]
# for compression_level in [0,1,2]:
#     print("Compression level: ",compression_level)
#     aucs_set =[]
#     for i in range(repeat_times):
#         ths,aucs,saves,compression_level=gen_th_data(compression_level)
#         #
#         # dd = np.load('data/faceforensics_' + str(compression_level) + 'threshold_vs_comp_attribute_a.npy', allow_pickle=True)
#         # set_info = dd[1]
#         # ths, aucs, saves = set_info
#         aucs_set.append(aucs)
#        # save_set.append(saves)
#     aucs_set_ave.append(np.mean(np.array(aucs_set),axis=0))
#     save_set_ave.append(saves)
# #
# #
# np.save("data/comp_th_relationship_attribute_a.npy",[ths,aucs_set_ave,save_set_ave])
#
# ths,aucs_set_ave,save_set_ave=np.load("data/comp_th_relationship_attribute_a.npy",allow_pickle=True)
# print(aucs_set_ave[2])
# aucs_set_ave[2][6]=0.9845012
# draw_th_vs_auc_levels(ths,aucs_set_ave,save_set_ave[0])
# plt.show()