import json

import numpy as np
import math
np.random.seed(2018)

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib
import matplotlib.pyplot as plt
def one_to_two(data):
    output=[]

    for i in range(int(len(data)/2)):
        output.append([data[2*i],data[2*i+1]])
    return output

def two_to_one(data):
    output=[]

    for i in range(len(data)):
        output.append(data[i][0])
        output.append(data[i][1])
    return output
def ml_analysis(data,label):
    X_train, X_test, y_train, y_test = train_test_split(np.array(data),np.array(label), test_size=0.3,random_state=17)
    # Ranodm Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=20)
    rf_clf.fit(X_train, y_train)
    rf_prediction_proba = rf_clf.predict_proba(X_test)[:, 1]
    print("Random forest")
    fpr, tpr, _, roc_auc = roc_curve_and_score(y_test, rf_prediction_proba)
    print("AUC:",roc_auc)
    getJudges(y_test,rf_prediction_proba,0.5)

def get_upper_lip_height(lip):
    sum=0
    for i in [2,3,4]:
        # distance between two near points up and down
        distance = math.sqrt( (lip[i][0] - lip[11+i][0])**2 +
                              (lip[i][1] - lip[11+i][1])**2   )
        sum += distance
    return sum / 3
def get_lower_lip_height(lip):
    sum=0
    for i in [8,9,10]:
        # distance between two near points up and down
        distance = math.sqrt( (lip[i][0] - lip[9+i][0])**2 +
                              (lip[i][1] - lip[9+i][1])**2   )
        sum += distance
    return sum / 3

def get_mouth_open(lip):
    sum=0
    for i in [13,14,15]:
        # distance between two near points up and down
        distance = math.sqrt( (lip[i][0] - lip[32-i][0])**2 +
                              (lip[i][1] - lip[32-i][1])**2   )
        sum += distance
    return sum / 3
def get_mouth_width(lip):
    return math.sqrt( (lip[12][0] - lip[16][0])**2 +
                              (lip[12][1] - lip[16][1])**2   )

def get_lip_shape_features(lip):
    return [get_mouth_open(lip),get_mouth_width(lip)]
def mergedata(datas,level='framelevel'):
    output = []
    if level=='framelevel':

        facesize = datas[0]
        facedistance = datas[1]
        compressionRatio = datas[2]

        for i in range(len(facesize)):
            for j in range(len(facesize[i])):
                for k in range(len(facedistance[i][j])):
                    output.append([facesize[i][j], facedistance[i][j][k], compressionRatio[i][j]])
    elif level=='videolevel':
        facesize = datas[0]
        facedistance = datas[1]
        compressionRatio = datas[2]
        for i in range(len(facesize)):
            for j in range(len(facesize[i])):

                output.append([facesize[i][j], np.average(np.array(facedistance[i][j])), compressionRatio[i][j]])

    return output
def loadDataSet(deepfake_data_path,authentic_data_path):  # 载入数据集

    dataMat_forgery = np.load(deepfake_data_path, allow_pickle=True)
    dataMat_compression = np.load(authentic_data_path, allow_pickle=True)

    dataMat=np.concatenate((dataMat_forgery,dataMat_compression),axis=0)#dataMat_forgery+dataMat_compression
   # print("dataMat_compression", len(dataMat),len(dataMat_compression),type(dataMat))
    labelMat=[1 for i in range(len(dataMat_forgery))]
    labelMat.extend([-1 for i in range(len(dataMat_compression))])
    return np.array(dataMat), np.array(labelMat)
def getAverage(data):
    sum=0
    for d in data:
        sum+=d
    return sum/len(data)
def getGaussian(data):
    mu = np.mean(data)  # 计算均值
    sigma = np.std(data)
    return mu,sigma
def mergedata_lip(datas,level='framelevel'):
    output = []
    if level=='framelevel':
        for i in range(len(datas)):
            for j in range(len(datas[i])):
                output.append(datas[i][j])


    elif level=='videolevel':
        for i in range(len(datas)):
            temp_lip_size=[]
            temp_distance=[]
        #    print(datas[i])
            for j in range(len(datas[i])):

                temp_lip_size.append(datas[i][j][0])
                temp_distance.append(datas[i][j][1])
           # print(getAverage(temp_lip_size),temp_lip_size)
          #  output.append([getAverage(temp_lip_size),getAverage(temp_distance)])
           # output.append([getAverage(temp_distance)])# test the correctness of mu
         #   output.append([max(temp_lip_size), getAverage(temp_distance)])#test method
            mu,sigma=getGaussian(temp_distance)
            output.append([max(temp_lip_size),mu,sigma])#test size area with sigma mu

    return output
def getObjectPaths(data_path,key_word):
    object_path=[]
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        file_path = file_path.replace('\\', '/')
        if key_word in file_path:
           # lip_info = np.load(video_path, allow_pickle=True)
            object_path.append(file_path)
    return object_path
def toAuthenData_lip(deepfake_data,authentic_data,level='framelevel'):  # 载入数据集

    dataMat_forgery=mergedata_lip(deepfake_data,level)
    dataMat_compression=mergedata_lip(authentic_data,level)

    dataMat=dataMat_forgery+dataMat_compression
   # print("dataMat_compression", len(dataMat),len(dataMat_compression),type(dataMat))
    labelMat=[1 for i in range(len(dataMat_forgery))]
    labelMat.extend([-1 for i in range(len(dataMat_compression))])
    return np.array(dataMat), np.array(labelMat)
def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return fpr, tpr, thresholds,roc_auc
def Find_Optimal_threshold(FPR,TPR,  thresholds):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = thresholds[Youden_index]

    return optimal_threshold
def getJudges(labels,scores,threshold):
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(len(scores)):

        if threshold < scores[i]:

            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
        else:

            if labels[i] == -1:
                tn += 1
            else:
                fn += 1
    getMetrics(tp, fp, tn, fn)
    return tp, fp, tn, fn
def getMetrics(tp, fp, tn, fn):
    precision=(tp / (tp + fp))
    recall=(tp / (tp + fn))
    F1=(2 * tp / (2 * tp + fp + fn))
    iou=(tp / (tp + fp + fn))
    accuracy=((tp + tn) / (tp + fp + fn + tn))


    print("Precision={:.2%}".format(precision))
    print("Recall={:.2%}".format(recall))
    print("F1={:.2%}".format(F1))
    print("IoU={:.2%}".format(iou))
    print("Accuracy={:.2%}".format(accuracy))

    return precision,recall,F1,iou,accuracy
def getMetricsNoPrint(tp, fp, tn, fn):
    precision=(tp / (tp + fp))
    recall=(tp / (tp + fn))
    F1=(2 * tp / (2 * tp + fp + fn))
    iou=(tp / (tp + fp + fn))
    accuracy=((tp + tn) / (tp + fp + fn + tn))




    return precision,recall,F1,iou,accuracy
def bootstrap(x, f, nsamples=1000):
    stats = [f(x[np.random.randint(x.shape[0], size=x.shape[0])]) for _ in range(nsamples)]
    return np.percentile(stats, (2.5, 97.5))


def bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5))


def permutation_test(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    idx1 = np.arange(X_train.shape[0])
    idx2 = np.arange(X_test.shape[0])
    auc_values = np.empty(nsamples)
    for b in range(nsamples):
        np.random.shuffle(idx1)  # Shuffles in-place
        np.random.shuffle(idx2)
        clf.fit(X_train, y_train[idx1])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test[idx2].ravel(), pred.ravel())
        auc_values[b] = roc_auc
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
    return roc_auc, np.mean(auc_values >= roc_auc)


def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)
def getAuc(labels, pred):
    '''将pred数组的索引值按照pred[i]的大小正序排序，返回的sorted_pred是一个新的数组，
       sorted_pred[0]就是pred[i]中值最小的i的值，对于这个例子，sorted_pred[0]=8
    '''
    sorted_pred = sorted(range(len(pred)), key=lambda i: pred[i])
    pos = 0.0  # 正样本个数
    neg = 0.0  # 负样本个数
    auc = 0.0
    last_pre = pred[sorted_pred[0]]
    count = 0.0
    pre_sum = 0.0  # 当前位置以前的预测值相等的rank之和，rank是从1开始的，因此在下面的代码中就是i+1
    pos_count = 0.0  # 记录预测值相等的样本中标签是正的样本的个数
    for i in range(len(sorted_pred)):
        if labels[sorted_pred[i]] > 0:
            pos += 1
        else:
            neg += 1
        if last_pre != pred[sorted_pred[i]]:  # 当前的预测几率值与前一个值不相同
            # 对于预测值相等的样本rank须要取平均值，而且对rank求和
            auc += pos_count * pre_sum / count
            count = 1
            pre_sum = i + 1  # 更新为当前的rank
            last_pre = pred[sorted_pred[i]]
            if labels[sorted_pred[i]] > 0:
                pos_count = 1  # 若是当前样本是正样本 ，则置为1
            else:
                pos_count = 0  # 反之置为0
        else:
            pre_sum += i + 1  # 记录rank的和
            count += 1  # 记录rank和对应的样本数，pre_sum / count就是平均值了
            if labels[sorted_pred[i]] > 0:  # 若是是正样本
                pos_count += 1  # 正样本数加1
    auc += pos_count * pre_sum / count  # 加上最后一个预测值相同的样本组
    auc -= pos * (pos + 1) / 2  # 减去正样本在正样本以前的状况
    auc = auc / (pos * neg)  # 除以总的组合数
    return auc
def get_area_mvs_strength(mvs_data,face_pos,margin=0):
    mvs_x, mvs_y=mvs_data
    print(len(mvs_x), len(mvs_y))
    index=0
   # print("face_pos",face_pos)
    left, top, right, bottom = face_pos
    pos_area_x=[]
    pos_area_y=[]
    while index<len(mvs_x):

        if mvs_y[index]<bottom+margin and mvs_y[index]>top-margin and mvs_x[index]<right+margin and mvs_x[index]>left-margin:
            pos_area_x.append(mvs_x[index])
            pos_area_y.append(mvs_y[index])
        index+=1
    return len(pos_area_x)
def get_area_mvs_strength2(mvs_data,face_pos,margin=0):
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

def get_descriptor_flag(mvs_data,face_areas,mvs_threshold=0,margin=10):
    flags=[]
    for face_pos in face_areas:
        if  get_area_mvs_strength(mvs_data,face_pos,margin)>mvs_threshold:
         #   print("mvs",get_area_mvs_strength(mvs_data,face_pos,margin))
            flags.append(True)
        else:
            #print("mvs", get_area_mvs_strength(mvs_data, face_pos, margin))
            flags.append(False)
    return flags
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
        mvs_vector_num,mvs_strength=get_area_mvs_strength2([pos_x,pos_y,d_x,d_y],face_pos,margin)
        if  mvs_strength>mvs_threshold:
            print("mvs",mvs_strength,mvs_path)
            flags.append(True)
        else:
            #print("mvs", get_area_mvs_strength(mvs_data, face_pos, margin))
            flags.append(False)
    return flags