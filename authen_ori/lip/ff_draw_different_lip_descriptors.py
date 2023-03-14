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
            if attr_threshold==0:
                selected_features.append(data[i])
                key_value=data[i][1][0][feature_id]
            else:
                if euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0] > attr_threshold:
                    # print("lip_attributes1[attr_i]",lip_attributes1[attr_i])
                    selected_features.append(data[i])
                    key_value = data[i][1][0][feature_id]

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
    #ori_pathes, fake_paths, crf_paths,mul_people_video = loadFeatureFiles(compression_level, '_features_all.npy')
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
def getDistribution(attr_i,compression_level):

    th_data = []
    aucs=[]
    save_set=[]
    fake_anomilies = [322, 725]
    if os.path.exists('data/faceforensics_comp'+str(compression_level)+'_descriptor' + str(attr_i) + '_distances.npy'):
        crf_distances, fake_distances = np.load(
            'data/faceforensics_comp'+str(compression_level)+'_descriptor' + str(attr_i) + '_distances.npy',
            allow_pickle=True)
    else:

        crf_distances, fake_distances, saves = collectDistances(attr_i, 0,
                                                                0, 0)
        np.save('data/faceforensics_comp'+str(compression_level)+'_descriptor' + str(attr_i) + '_distances.npy',
            [ crf_distances, fake_distances])

    fake_distances = clear_anomilies(fake_distances, fake_anomilies)
    show_distribution(crf_distances, fake_distances,
                       "figs/comp"+str(compression_level)+"_lip_descriptor_dist_"+str(attr_i)+".pdf")

    crf_distances, fake_distances = np.array(crf_distances), np.array(fake_distances)

    roc_auc, accuracy, recall = getAuthenPerformance(crf_distances, fake_distances)

    aucs.append(roc_auc)

    return crf_distances, fake_distances
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
    fig=plt.figure()
    ax =fig.add_subplot(111)
    plt.scatter([d[0] for d in crf_score],[d[1] for d in crf_score],label='Authentic',alpha=0.5)
    plt.scatter([d[0] for d in fake_score], [d[1] for d in fake_score], label='Lip-Reenactment',alpha=0.5)
   # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.xticks(fontsize=21)
    plt.xlabel(r'$\mu$',fontsize=21)
    plt.ylabel(r'$\sigma$',fontsize=21)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.3))
    plt.yticks(fontsize=21)
   # plt.margins(0,0)
   # plt.xlim((-0.01,0.64))
   # plt.ylim((-0.01,0.455))
    plt.legend(fontsize=21)
    plt.savefig(name,bbox_inches='tight')

  #  plt.show()
def rename_file(compression_level):
    ths = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
   # ths=[0.35]
    th_data = []
    i = 0
    fixed_th=0.4
    aucs=[]
    save_set=[]
    for th in ths:
        old_name='data/faceforensics_comp'+str(compression_level)+'_threshold' + str(i) + '_distances_npy_cur_all.npy'
        new_name='data/faceforensics_comp'+str(compression_level)+'_threshold' + str(i) + '_distances_'+'false_refpoint.npy'
        os.rename(old_name,new_name)
        i+=1
   # plt.show()
def draw_th_vs_auc_levels(ths,aucs_set,saving):
    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_subplot(111)
    lns1 = ax.plot(ths, aucs_set[0], color='red',marker='o', linestyle='solid', label='AUC-crf20')
    lns2 = ax.plot(ths, aucs_set[1], color='green',marker='v', linestyle='solid', label='AUC-crf30')
    lns3 = ax.plot(ths, aucs_set[2], color='blue',marker='s', linestyle='solid', label='AUC-crf40')
    ax.set_ylim(0.94, 1.)
    # ax.legend(loc=0,fontsize=14)
    ax.grid()
    ax.set_xlabel(r'$t_s$', fontsize=16)
    ax.set_ylabel('AUC', fontsize=16)
   # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.tick_params(labelsize=16)
    # ax.set_yticks(fontsize=14)
    ax2 = ax.twinx()
   # print(saving)
    lns4 = ax2.plot(ths, [1 - d for d in saving], color='orange',marker='^', linestyle='solid', label=r'$1-SR$')
    ax2.tick_params(labelsize=16)

    ax2.set_ylabel(r'$1-SR$', fontsize=16)
    lns = lns1 + lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="lower right", fontsize=16)

    plt.savefig("figs/lip_comps_th_vs_auc.pdf",bbox_inches='tight')
    plt.show()
def getGuassian(distances):
    distances_video=[]
    for d in distances:
        distances_video.append(np.mean(d))
    return np.mean(distances_video), np.std(distances_video)
aucs_set=[]
compression_levels=[0,1,2]
save_set=[]
for attr_i in [0,1,2,3]:
    print("attr_i ",attr_i)

    real_attributes,fake_attributes=getDistribution(attr_i,0)
    #print(fake_attributes)
    guassian_real,guassian_fake=getGuassian(real_attributes),getGuassian(fake_attributes)
    print("cfadvgds",guassian_real,guassian_fake)
    plt.figure()

    means_real,means_fake=[],[]
    for d in real_attributes:
        means_real.append(math.sqrt(np.mean(d)))
    for d in fake_attributes:
        means_fake.append(math.sqrt(np.mean(d)))
    print(getGuassian(means_real), getGuassian(means_fake))
    bins=np.linspace(0, 1, 30)
    plt.hist(means_real,bins)
    plt.hist(means_fake,bins)
    plt.show()
   # np.save('data/faceforensics_descriptor'+str(attr_i)+'_distribution.npy', [real_attributes,fake_attributes])
