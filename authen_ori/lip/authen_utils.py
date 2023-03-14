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
def get_samples(data,label,threshold):
  #  print("Test Overall")
    tp,tn,fp,fn=0,0,0,0
    for i in range(len(data)):
        if data[i]>threshold:
            if label[i]==1:
                tp+=1
            else:
                fn+=1
        else:
            if label[i]==1:
                fp+=1
            else:
                tn+=1
    accuracy=(tp+tn)/(tp+fp+tn+fn)
    recall=(tp)/(tp+fn)
    return tp,fp,tn,fn
def get_lip_attributes(lip_landmarks, lip_size=None):
    if lip_size is None:
        lip_size=[1,1]

    d1,d2,d3,d4=[],[],[],[]
    d1.append((lip_landmarks[6][0]-lip_landmarks[0][0])/lip_size[0])#width
    d1.append((lip_landmarks[9][1] - lip_landmarks[3][1])//lip_size[1])#height
    for i in range(1,6):
        d2.append((lip_landmarks[12-i][1] - lip_landmarks[i][1])//lip_size[1])
    center_point=[((lip_landmarks[6][0]+lip_landmarks[0][0])/2),(lip_landmarks[9][1] + lip_landmarks[3][1])/2]

    for i in range(12):

        d3.append(euclidean_distance([[lip_landmarks[i][0]/lip_size[0],lip_landmarks[i][1]/lip_size[1]]],[[center_point[0]/lip_size[0],center_point[1]/lip_size[1]]])[0,0])
    for i in range(12,20):
        d4.append(euclidean_distance([[lip_landmarks[i][0] / lip_size[0], lip_landmarks[i][1] / lip_size[1]]],
                                     [[center_point[0] / lip_size[0], center_point[1] / lip_size[1]]])[0, 0])
   # print("d1,d2,d3,d4",d1,d2,d3,d4)
    return d1,d2,d3,d4

#只根据mvs info传送lip 数据。   （1）接收方可以测试d1，d2，d3，d4分别对验证结果的影响 以及和landmarks直接的对比
#测试和landmarks差距选择的区别
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
        face_info = FaceFeature([0,1,2,3,4], video_data[i][1], [100, 50, 150, 100], i)
        match_result = persons._match([face_info],i, [])

    return persons

def pick_KeyLips_mvs(video_path,persons,mvs_threshold):

    new_persons=FaceTracker()
    save_comm=[]
    filename, t = os.path.splitext(video_path)
    if '_v1_dlib_features' in filename:
        filename = filename.replace('_v1_dlib_features', '')
    elif '_lip_all_info_v0' in filename:
        filename = filename.replace('_lip_all_info_v0', '')


   # print("HHHHHHH")
    for face_track in persons.tracks:
        new_p = FaceTrack(face_track.face_id, face_track._n_init, face_track._max_age, face_track.start,
                           face_track.features[0])
        new_p.state=face_track.state

        for feature in face_track.features:
            frame_pos=feature.frame_pos
            mvs_path = filename + "/" + str(frame_pos + 1) + "_mvs.json"
            mvs_data=get_mvs_data(mvs_path)
            #print(get_area_mvs_strength(mvs_data, feature.face_area,mvs_threshold))
            x_mvs_len,mvs_strength=get_area_mvs_strength(mvs_data, feature.face_area,mvs_threshold)
            if mvs_strength>mvs_threshold:
                new_p.features.append(feature)
        new_persons.tracks.append(new_p)
        new_persons._next_id += 1
        save_comm.append(len(new_p.features)/len(face_track.features))
    return new_persons, getAverage(save_comm)
#根据选中的属性选择是否传输lip 数据。   接收方可以测试d1，d2，d3，d4分别对验证结果的影响 以及和landmarks直接的对比
def pickKeyLips_attribute(persons, attr_arr,attr_threshold):
    new_persons = FaceTracker()
    save_comm = []
    for face_track in persons.tracks:
        new_p = FaceTrack(face_track.face_id, face_track._n_init, face_track._max_age, face_track.start,
                           face_track.features[0])
        new_p.state=face_track.state

        for feature in face_track.features:
            lip_attributes1 = get_lip_attributes(new_p.features[-1].lip_descriptor)
            lip_attributes2 = get_lip_attributes(feature.lip_descriptor)
            frame_lip_distance=[]
            for attr_i in attr_arr:
                frame_lip_distance.append( euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0])
            if getAverage(frame_lip_distance) > attr_threshold:
                new_p.features.append(feature)
        new_persons.tracks.append(new_p)
        new_persons._next_id += 1
        save_comm.append(len(new_p.features) / len(face_track.features))
    return new_persons, getAverage(save_comm)
def compare_distance_lip(p1, p2,lip_size=None, len_prob=1):
    #p1 waiting for authentication
    #p2 stored info
    #len_prob=len(p1.landmarks)/len(p2.landmarks)---ori/cur
    lip_distance = []
    i = 0
    j = 0
    count=0
    while i < len(p1.features) and j < len(p2.features) - 1:
     #   print("pos",p1.features[i].frame_pos,p2.features[j + 1].frame_pos)
        if p1.features[i].frame_pos < int(len_prob*(p2.features[j + 1].frame_pos)):
            lip_attributes1=get_lip_attributes(p1.features[i].lip_descriptor,lip_size)
            lip_attributes2=get_lip_attributes(p2.features[j].lip_descriptor,lip_size)
          #  print("i,j",p1.features[i].frame_pos,p2.features[j ].frame_pos)
            i += 1
        else:
            if j == len(p2.features) - 2:
                lip_attributes1 = get_lip_attributes(p1.features[i].lip_descriptor,lip_size)
                lip_attributes2 = get_lip_attributes(p2.features[j+1].lip_descriptor,lip_size)
               # print("i,j",p1.features[i].frame_pos,p2.features[j ].frame_pos)
            j += 1
            continue
        frame_lip_distance=[]
        count+=1
      #  print("lip_attributes1",len(lip_attributes1))
        for attr_i in range(len(lip_attributes1)):
            frame_lip_distance.append(
                euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0])
       # print("frame_lip_distance",frame_lip_distance)
        lip_distance.append(frame_lip_distance)
   # print("lip_distance",count,len(lip_distance))
    return lip_distance

def get_lip_size(data):
    max=np.max(data,axis=0)
    print("max",max)

def get_authen_data(cur_faces, stored_faces,lip_size=None,len_prob=1):
    matched_id = match_iou(cur_faces, stored_faces)
    #print("matched success", matched_id)
    persons_distance = []
    persons_ids = []
    index = 0
    for facetrack in cur_faces.tracks:
        print("matched success",len(facetrack.features),len(stored_faces.tracks[matched_id[index]].features))
        person_distance = compare_distance_lip(facetrack, stored_faces.tracks[matched_id[index]],lip_size,len_prob)
        persons_distance.append(person_distance)
        persons_ids.append(facetrack.face_id)
        index += 1
    return persons_distance, persons_ids


def pick_attribute(statistic_data,attribute_arr,add_std=True):
   # print(statistic_data)
    out=[]
    for d in statistic_data:
        d_out=[]
        for i in attribute_arr:
            d_out.append(d[i][0])#add means
            if add_std:
                d_out.append(d[i][1])  # add std
        out.append(d_out)
    return out
def draw_attributes(data1,data2,attr_i,compression_level,title='',std_flag=False):
    if std_flag:
        plt.figure()
        plt.scatter([d[attr_i][0] for d in data1], [d[attr_i][1] for d in data1],label='deepfake',alpha=0.5)
        plt.scatter([d[attr_i][0] for d in data2], [d[attr_i][1] for d in data2],label='authentic',alpha=0.5)
        plt.xlabel("mu")
        plt.ylabel('sigma')
        plt.title(title)
        plt.legend()
        plt.savefig("figs/complevel"+str(compression_level)+"attribute_std_" + str(attr_i) + ".png")
        plt.show()
    else:
        plt.figure()
        plt.hist([d[attr_i][0] for d in data1],bins=40,range=(0,1),alpha=0.5)
        plt.hist([d[attr_i][0] for d in data2], bins=40, range=(0, 1),alpha=0.5)
        plt.title(title)
        plt.savefig("figs/complevel"+str(compression_level)+"_attribute"+str(attr_i)+".png")
        plt.show()



