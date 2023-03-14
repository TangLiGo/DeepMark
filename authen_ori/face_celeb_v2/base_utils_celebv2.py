import face_rec_video as fc
import os
import cv2
import numpy as np
import dlib
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
from PIL import Image
import os.path as osp
from scipy.signal import find_peaks
import json
import face_rec_video as fc
import matplotlib.pyplot as plt
face_rec_model_path = "C:/Users/Tangli/Downloads/codes/dlib_face_recognition_resnet_model_v1.dat"
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
detector = dlib.get_frontal_face_detector()
predictor_path = "C:/Users/Tangli/Downloads/codes/shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(predictor_path)
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
def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return fpr, tpr, thresholds,roc_auc
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
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=None)
    ths=[0.05,0.1,0.15,0.2,0.25,0.35,0.4,0.45,0.5,0.55,0.6]
    tprs=[]
    fprs=[]
    for th in ths:
        tpr,fpr=get_roc_data([X[0] for X in X_train], y_train,th)
        tprs.append(tpr)
        fprs.append(fpr)
    th = Find_Optimal_threshold(np.array(fprs), np.array(tprs), ths)
   # print("y_test",[X[0] for X in X_test])
    accuracy, recall = compute_accuracy(y_test,[X[0] for X in X_test],  th)
    return th,accuracy, recall
def ml_analysis(data,label,print_false_samples=False):
    X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.3,random_state=None)#np.random.randint(1,200,1)[0]
  #  print("ytest",X_test,y_test)
    # Ranodm Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=15)

    rf_clf.fit(X_train, y_train)
    rf_prediction_proba = rf_clf.predict_proba(X_test)[:, 1]
#
  #  print("Random forest")
    fpr, tpr, _, roc_auc = roc_curve_and_score(np.array(y_test), rf_prediction_proba)
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
def check_multi_people(data,th=0.6):
    distances=[]
    flag=False
    for i in range(len(data)-1):
        if len(data[i+1][1])==0 or len(data[i][1])==0:
            continue
        dis=fc.getDistance(data[i][1][0][0],data[i+1][1][0][0])
        distances.append(dis)
        if dis>th:
            flag=True

    return distances,flag
def getFolderFiles(folder_path,key_word='_features_all_track.npy'):
    pathes=[]

    for file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, file)
        video_path = video_path.replace('\\', '/')

        if '.mp4' in video_path:
            p1 = os.path.basename(video_path)
            file_name = os.path.splitext(p1)[0]
            object_path=folder_path+'/'+file_name+key_word#'_features_all_track.npy'

            pathes.append(object_path)

    return pathes


def divide_fake_paths(ori_paths,fake_paths):
    fake_sets=[]
    for i in range(len(ori_paths)):
        sub_fake_paths=[]
        p1 = os.path.basename(ori_paths[i])
        file_name = (os.path.splitext(p1)[0]).replace('_features_all_track', '')
        d=str.split(file_name,'_')
       # print('ori',file_name)
        for path in fake_paths:
            if d[0] in path and d[1] in path and file_name not in path:
                sub_fake_paths.append(path)
               # print("fake",path)
        fake_sets.append(sub_fake_paths)
    return fake_sets

def loadFeatureFiles(compression_level=0,dataset='Celeb-v2',key_word='_features_all_track.npy'):
    data_path_real = 'D:/TangLi/datasets/Celeb-DF-v2/Celeb-real'
    data_path_com1 = 'D:/TangLi/datasets/Celeb-DF-v2/Celeb-real-crf20'
    data_path_com2 = 'D:/TangLi/datasets/Celeb-DF-v2/Celeb-real-crf30'
    data_path_com3 = 'D:/TangLi/datasets/Celeb-DF-v2/Celeb-real-crf40'

    data_path_forged1 = 'D:/TangLi/datasets/Celeb-DF-v2/Celeb-synthesis'
   # data_path_forged2 = 'D:/TangLi/datasets/Celeb-DF-v2/Celeb-synthesis_crf20'
    data_path_forged3 = 'D:/TangLi/datasets/Celeb-DF-v2/Celeb-synthesis-crf30'
    data_path_forged4 = 'D:/TangLi/datasets/Celeb-DF-v2/Celeb-synthesis-crf40'
    data_path_com = [data_path_com1, data_path_com2, data_path_com3]
    data_path_forged=[data_path_forged1,data_path_forged3,data_path_forged4]
    key_word = '_features_all_track.npy'
    multi_person_v=np.load('celeb_v2_multi_person_video.npy', allow_pickle=True)
    ori_paths=getFolderFiles(data_path_real,key_word)
    crf_paths = getFolderFiles(data_path_com[compression_level], key_word)
    fake_paths_raw = getFolderFiles(data_path_forged[compression_level], key_word)
    fake_paths=divide_fake_paths(ori_paths,fake_paths_raw)

    #clear mul person_videos
    return ori_paths,fake_paths,crf_paths,multi_person_v

