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

def euclidean_distance(a, b):
    # 用于计算成对的平方距离
    # a NxM 代表N个对象，每个对象有M个数值作为embedding进行比较
    # b LxM 代表L个对象，每个对象有M个数值作为embedding进行比较
    # 返回的是NxL的矩阵，比如dist[i][j]代表a[i]和b[j]之间的平方和距离
    # 实现见：https://blog.csdn.net/frankzd/article/details/80251042
    a, b = np.asarray(a), np.asarray(b)  # 拷贝一份数据
    if len(a) == 0 or len(b) == 0:
       # #print("ec", len(np.zeros((len(a), len(b)))))
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(
        b).sum(axis=1)  # 求每个embedding的平方和
    # sum(N) + sum(L) -2 x [NxM]x[MxL] = [NxL]
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    #返回的矩阵 每一行i对应的是a[i] 与各个b[j] j=1-m 的距离

    return np.sqrt(r2)
def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return fpr, tpr, thresholds,roc_auc
def compute_accuracy(y_true, y_pred,th):
   # print("y_pred",th,y_pred)
    correct_predictions = 0
    # iterate over each label and check
    tps,fps,tns,fns=0,0,0,0
    for true, predicted in zip(y_true, y_pred):
        if true ==1 and predicted>=th:
            tps += 1
        elif true==1 and predicted<th:
            fns+=1
        elif true==-1 and predicted>=th:
            fps+=1
        elif true==-1 and predicted<th:
            tns+=1

    accuracy=(tps+tns)/(tps+tns+fps+fns)
    recall=(tps)/(tps+fns)
    # compute the accuracy

    return accuracy,recall
def get_perfect_th(fpr, tpr,thresholds):
 #   print(fpr,tpr,thresholds)
    for i in range(len(tpr)):
        if tpr[i]>0.99:
            return thresholds[i],fpr[i]
def Find_Optimal_threshold(FPR,TPR,  thresholds):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = thresholds[Youden_index]

    return optimal_threshold
def get_roc_data(data,label,threshold):
    tps,fps,tns,fns=0,0,0,0
    #print(data)
    for true, predicted in zip(label, data):
        if true ==1 and predicted>=threshold:
            tps += 1
        elif true==1 and predicted<threshold:
            fns+=1
        elif true==-1 and predicted>=threshold:
            fps+=1
        elif true==-1 and predicted<threshold:
            tns+=1
   # print(tps,fps,tns,fns)
    return tps/(tps+fns),fps/(fps+tns)
def threshold_analysis(data,label):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=12)
    ths=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.25,0.3,0.35]
    tprs=[]
    fprs=[]
    for th in ths:
        tpr,fpr=get_roc_data([X[0] for X in X_train], y_train,th)
        tprs.append(tpr)
        fprs.append(fpr)
    th = Find_Optimal_threshold(np.array(fprs), np.array(tprs), ths)
    accuracy, recall = compute_accuracy(y_test,[X[0] for X in X_test],  th)
    return th,accuracy, recall
def ml_analysis(data,label,print_false_samples=False):
    X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.3,random_state=12)#np.random.randint(1,200,1)[0]
  #  print("ytest",X_test,y_test)
    # Ranodm Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=15)

    rf_clf.fit(X_train, y_train)
    rf_prediction_proba = rf_clf.predict_proba(X_test)[:, 1]
#
  #  print("Random forest")
    fpr, tpr, _, roc_auc = roc_curve_and_score(np.array(y_test), rf_prediction_proba)
    _, _, _, roc_auc = roc_curve_and_score(np.array(label), np.array(data))
    # plt.figure()
    # plt.plot(fpr,tpr,'rx')
    # plt.show()
    threshold = Find_Optimal_threshold(fpr, tpr, _)
  # print("fpr, tpr:",fpr, tpr)
    accuracy,recall=compute_accuracy(y_test,rf_prediction_proba,threshold)

 #   print(" threshold range:{}".format(_))
    #th,fp=get_perfect_th(fpr, tpr, _)
   # print("th:{},fpr:{}".format(th,fp))
   # print()
   #  plt.figure()
   #  plt.plot(fpr,tpr,'rx')
   #  plt.show()
  #  print("AUC:",roc_auc)
    if print_false_samples:
        fps = []
        fns = []
        for idx, prediction, label in zip(enumerate(X_test), rf_prediction_proba, y_test):

            if label == 1 and prediction < threshold:
                fns.append(idx[0])
                print("Sample", idx, ', has been classified as', prediction, 'and should be', label)
            if label == -1 and prediction >= threshold:
                fps.append(idx[0])
                print("Sample", idx, ', has been classified as', prediction, 'and should be', label)
            # if idx[0]==163:
            #     print('get score',prediction,"Sample", idx, ', has been classified as', prediction, 'and should be', label)
        ori_fps = []
        ori_fns = []
       # print(fps,fns)
        for i in range(len(data)):
            for j in fps:
                if (np.array(data[i]) == np.array(X_test[j])).all():
                    ori_fps.append(i)
            for j in fns:
                if (np.array(data[i]) == np.array(X_test[j])).all():
                    ori_fns.append(i)
        # accuracy, recall=compute_accuracy(y_test,rf_prediction_proba,0.5)
        # print("accuracy={},recall={}".format(accuracy, recall))
    return roc_auc,accuracy,recall
def getAverage(data):
    return np.mean(np.array(data))


def _cosine_distance(a, b, data_is_normalized=False):
    # a和b之间的余弦距离
    # a : [NxM] b : [LxM]
    # 余弦距离 = 1 - 余弦相似度
    # https://blog.csdn.net/u013749540/article/details/51813922
    if not data_is_normalized:
        # 需要将余弦相似度转化成类似欧氏距离的余弦距离。
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        #  np.linalg.norm 操作是求向量的范式，默认是L2范式，等同于求向量的欧式距离。
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)
def intersect_area(a,b):
    x1=max(a[0],b[0])
    x2=max(a[1],b[1])
    x3=min(a[2],b[2])
    x4=min(a[3],b[3])
    return area([x1,x2,x3,x4])

def area(a):
   # print("area",a,type(a))
    h=a[3]-a[1]
    w=a[2]-a[0]
    if w < 0 or h < 0: return 0
    return w * h
def get_IoUs(areas_new,areas_old):
   # a, b = np.asarray(areas_old), np.asarray(areas_new)
    out=np.zeros((len(areas_new), len(areas_old)))
    for i, new in enumerate(areas_new):
        for j,old in enumerate(areas_old):
            out[i,j]=intersect_area(old,new)/(area(new)+area(old)-intersect_area(old,new))
    return out
def match_iou(cur_faces,stored_faces,metric='euclidean'):

    face_ious = get_IoUs([facetrack.features[0].face_area for facetrack in cur_faces.tracks], [facetrack.features[0].face_area for facetrack in stored_faces.tracks])

    sorted_iou_indices = np.argsort(-face_ious, axis=1)
   # print(cur_faces._next_id,sorted_iou_indices)
    matched_id=[sorted_iou_indices[id][0] for id in range(cur_faces._next_id)]
    #print("face_ious", matched_id)
    return matched_id

def getStatistics(data):
    attribute_num=len(data[0])
    data=np.array(data)
    out=[]

    std_out=np.std(data,axis=0)

    mean_out = np.mean(data, axis=0)
    for i in range(attribute_num):
        out.append([mean_out[i],std_out[i]])

    return out
def get_main_person(faces):
    main_person_id=0
    length=len(faces.tracks[0].features)
    for id in range(faces._next_id):
        if length<len(faces.tracks[id].features):
            main_person_id=id
        if length==len(faces.tracks[id].features):
            if area(faces.tracks[id].features[0].face_area)>area(faces.tracks[main_person_id].features[0].face_area):
                main_person_id = id
                print("here")
    return main_person_id

def readData(data_path):
    f=open(data_path,"rb")

    while 1:
        try:
            persons = pickle.load(f)

            return persons
        except:
            break
def getObjectPaths(data_path,key_word):
    object_path=[]
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        file_path = file_path.replace('\\', '/')

        if key_word in file_path :#and '_track' not in file_path:
            # print("file_path", file_path)
           # lip_info = np.load(video_path, allow_pickle=True)
            object_path.append(file_path)
    return object_path

def traverse_celeb_videos(compression_level=0):
    data_path_real = 'D:/TangLi/datasets/Celeb-DF/Celeb-real'
    data_path_com1 = 'D:/TangLi/datasets/Celeb-DF/Celeb-real_crf20'
    data_path_com2 = 'D:/TangLi/datasets/Celeb-DF/Celeb-real_crf30'
    data_path_com3 = 'D:/TangLi/datasets/Celeb-DF/Celeb-real_crf40'
    data_path_forged0 = 'D:/TangLi/datasets/Celeb-DF/Celeb-synthesis'
    data_path_forged1 = 'D:/TangLi/datasets/Celeb-DF/Celeb-synthesis_crf20'
    data_path_forged2 = 'D:/TangLi/datasets/Celeb-DF/Celeb-synthesis_crf30'
    data_path_forged3 = 'D:/TangLi/datasets/Celeb-DF/Celeb-synthesis_crf40'
    data_path_com = [data_path_com1, data_path_com2, data_path_com3]
    data_path_forged = [data_path_forged0,data_path_forged2,data_path_forged3]
    original_videos=getObjectPaths(data_path_real,'features_all')
   # print("original_videos",original_videos)
    compressed_videos =getObjectPaths(data_path_com[compression_level],'features_all')

    judgements_forgery = []
    for i in range(len(original_videos)):
        p1 = os.path.basename(original_videos[i])
        folder_name = (os.path.splitext(p1)[0]).replace('_lip_all_info_v0','')
        forged_path = data_path_forged[0]+'/' + folder_name
        judgements_forgery.append(getObjectPaths(forged_path,'_lip_all_info_v0'))


    return original_videos,judgements_forgery,compressed_videos
def traverse_forensics_videos2(compression_level=0):
    data_path_real = 'D:/TangLi/datasets/FaceForensics++/original_sequences/youtube/c23/videos'
    data_path_com1 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf20'
    data_path_com2 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf30'
    data_path_com3 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf40'

    data_path_forged0 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf20'
    data_path_forged1 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf30'
    data_path_forged2 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf40'
    data_path_com = [data_path_com1, data_path_com2, data_path_com3]
    data_path_forged=[data_path_forged0,data_path_forged1,data_path_forged2]
    original_lip_info_path=getObjectPaths(data_path_real,'features_all_track')
    forged_lip_info_path=getObjectPaths(data_path_forged[compression_level],'features_all_track')
    crf_lip_info_path=getObjectPaths(data_path_com[compression_level],'features_all_track')
  #  print("forged_lip_info_path",forged_lip_info_path)
    return original_lip_info_path,forged_lip_info_path,crf_lip_info_path
def traverse_forensics_videos3(compression_level=0):
    data_path_real = 'D:/TangLi/datasets/FaceForensics++/original_sequences/youtube/c23/videos'
    data_path_com1 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf20'
    data_path_com2 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf30'
    data_path_com3 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf40'

    data_path_forged0 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf20'
    data_path_forged1 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf30'
    data_path_forged2 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf40'
    data_path_com = [data_path_com1, data_path_com2, data_path_com3]
    data_path_forged=[data_path_forged0,data_path_forged1,data_path_forged2]
    original_lip_info_path=getObjectPaths(data_path_real,'features_all.npy')
    forged_lip_info_path=getObjectPaths(data_path_forged[compression_level],'features_all.npy')
    crf_lip_info_path=getObjectPaths(data_path_com[compression_level],'features_all.npy')
  #  print("forged_lip_info_path",forged_lip_info_path)
    return original_lip_info_path,forged_lip_info_path,crf_lip_info_path
def traverse_forensics_videos_key(compression_level=0,keyword='features_all.npy'):
    data_path_real = 'D:/TangLi/datasets/FaceForensics++/original_sequences/youtube/c23/videos'
    data_path_com1 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf20'
    data_path_com2 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf30'
    data_path_com3 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf40'

    data_path_forged0 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf20'
    data_path_forged1 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf30'
    data_path_forged2 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf40'
    data_path_com = [data_path_com1, data_path_com2, data_path_com3]
    data_path_forged=[data_path_forged0,data_path_forged1,data_path_forged2]
    original_lip_info_path=getObjectPaths(data_path_real,keyword)

    return original_lip_info_path
def traverse_forensics_videos(compression_level=0):
    data_path_real = 'D:/TangLi/datasets/FaceForensics++/original_sequences/youtube/c23/videos'
    data_path_com1 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf20'
    data_path_com2 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf30'
    data_path_com3 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf40'

    data_path_forged0 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf20'
    data_path_forged1 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf30'
    data_path_forged2 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf40'
    data_path_com = [data_path_com1, data_path_com2, data_path_com3]
    data_path_forged=[data_path_forged0,data_path_forged1,data_path_forged2]
    original_lip_info_path=getObjectPaths(data_path_real,'features_all.npy')
    forged_lip_info_path=getObjectPaths(data_path_forged[compression_level],'lip_all_info')
    crf_lip_info_path=getObjectPaths(data_path_com[compression_level],'features_all.npy')
  #  print("forged_lip_info_path",forged_lip_info_path)
    return original_lip_info_path,forged_lip_info_path,crf_lip_info_path
def get_descriptor_flag_by_path(mvs_path,face_areas,mvs_threshold=0,margin=10):
    with open(mvs_path) as input_file:
        mvs_data = json.load(input_file)
    index = 0

    pos_x = []
    pos_y = []
    d_x=[]
    d_y=[]
    while index < len(mvs_data):
        item = mvs_data[index]
        pos_x.append(item['dst_x'])
        pos_y.append(item['dst_y'])
        d_x.append(item['dx'])
        d_y.append(item['dy'])
        index += 1
    flags=[]
    for face_pos in face_areas:
        mvs_vector_num,mvs_strength=get_area_mvs_strength([pos_x,pos_y,d_x,d_y],face_pos,margin)
        if  mvs_strength>mvs_threshold:

            flags.append(True)
        else:
            flags.append(False)
    return flags
def get_area_mvs_strength(mvs_data,face_pos,margin=0):
    mvs_x, mvs_y, d_x, d_y=mvs_data
    strength=0
    index = 0
    # print("face_pos",face_pos)
    left, top, right, bottom = face_pos
    pos_area_x = []
    pos_area_y = []
    while index < len(mvs_x):
        if mvs_y[index] < bottom + margin and mvs_y[index] > top - margin and mvs_x[index] < right + margin and mvs_x[
            index] > left - margin:
            pos_area_x.append(mvs_x[index])
            pos_area_y.append(mvs_y[index])
            strength+=abs(d_x[index])
            strength+abs(d_y[index])
         #   print("dx,dy",d_x[index],d_y[index],mvs_x[index],mvs_y[index])
        index += 1
    return len(pos_area_x),strength
def get_mvs_data(mvs_path):
    with open(mvs_path) as input_file:
        mvs_data = json.load(input_file)
    index = 0

    pos_x = []
    pos_y = []
    d_x=[]
    d_y=[]
    while index < len(mvs_data):
        item = mvs_data[index]

        pos_x.append(item['dst_x'])
        pos_y.append(item['dst_y'])
        d_x.append(item['dx'])
        d_y.append(item['dy'])
        index += 1
    return [pos_x,pos_y,d_x,d_y]
def write_excel_xls(path,sheet_name,value):

    index=len(value)
    workbook=xlwt.Workbook()
    sheet=workbook.add_sheet(sheet_name)
    for i in range(0,index):
        for j in range(0,len(value[i])):
            sheet.write(i,j,value[i][j])
    workbook.save(path)
    print("writing success xls ")