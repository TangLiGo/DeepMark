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
def mul_people(detect_data):
    for i in range(len(detect_data)):
        if len(detect_data[i][1])>1:
            return True
    return False
def getAreaIoUs(detect_data,track_data,opt=0):
    ious=[]
    mis_times=0
    win_times=0
    for i in range(len(detect_data)):
        if len(detect_data[i][1]) == 0 or len(track_data[i][2]) == 0:
            continue
        face_ious = get_IoUs([face[1] for face in detect_data[i][1]],
                             [face[1] for face in track_data[i][2]])

        match_ious=face_ious.max(axis=0)[:min(len(detect_data[i][1]),len(track_data[i][2]))]

      #  print(len(detect_data[i][1]), len(track_data[i][2]), match_ious,face_ious)

        if len(detect_data[i][1])>len(track_data[i][2]):
            mis_times+=1
        elif len(detect_data[i][1])<len(track_data[i][2]):
            win_times+=1

        # for iou in match_ious:
        #     ious.append(iou)
        ious.append(np.max(face_ious))
      #  ious.extend(match_ious[:min(len(detect_data[i][1]),len(track_data[i][2]))])

    return ious,mis_times,win_times
def collectDetectTimes(track_data):
    times=0
    for d in track_data:
        if d[1]==True:
            times+=1
    return times/len(track_data)

def collectDistances(compression_level,track_key,detect_key):
    ori_track_pathes= traverse_forensics_videos_key(compression_level,track_key)
    ori_pathes = traverse_forensics_videos_key(compression_level,detect_key)

    ids=[447]#[447,509,566]#(0.3,0.4)

    ious_collect=[]
    mis_times_collect,win_times_collect=[],[]
    id=0
    track_props=[]
   # mul_record=np.load('data/origin_mul_people.npy',allow_pickle=True)
    mul_record = np.load("data/faceforensics_multi_person_video_face2face.npy", allow_pickle=True)
   # print(mul_record)
    for i in range(len(ori_pathes)):


        # if mul_people(np.load(ori_pathes[i], allow_pickle=True)):
        #     mul_record.append(ori_pathes[i])
        #     continue
        # if ori_pathes[i] in mul_record:
        #     continue
        # print(ori_pathes[i], ori_track_pathes[i])
        # if '908' in ori_pathes[i]:
        #     print(id)


        ious,mis_times,win_times = getAreaIoUs(np.load(ori_pathes[i], allow_pickle=True),
                           np.load(ori_track_pathes[i], allow_pickle=True))
        if np.min(ious)<0.3 and np.min(ious)>0.0:
            print("recoffff",id,ori_pathes[i])
            print(ious)
        ious_collect.append(ious)
        mis_times_collect.append(mis_times)
        win_times_collect.append(win_times)
        track_props.append(collectDetectTimes(np.load(ori_track_pathes[i], allow_pickle=True)))
        # if id in ids:
        #     #  print(np.load(ori_pathes[i],allow_pickle=True))
        #     print(ious)
        #     plt.figure()
        #     plt.hist(ious)
        #     plt.show()
        id += 1

    # #
    # draw_frame_dis(ious_collect)
    # draw_video_dis(ious_collect)
    # draw_video_dis_2d(ious_collect)
   #print(fake_distances)
 #   np.save('data/origin_mul_people.npy',mul_record)
    return ious_collect,mis_times_collect,win_times_collect,track_props

def draw_frame_dis(ious_collect,name='frame_dist'):
    frame_ious=[]
    id_v=0

    for ious in ious_collect:
        frame_ious.extend(ious)
        id_f = 0
        for iou in ious:
            if iou>0 and iou<0.1:
                print("id",id_v,id_f)
                break
            id_f+=1
        id_v+=1

    plt.figure()
    tt=plt.hist(frame_ious)
    print(tt)
    plt.savefig('figs/'+name)
    #plt.show()
def draw_frame_dis_collect(ious_collect_detection_free_track,ious_collect_proposed_timer,ious_collect_proposed_notimer):
    frame_ious1,frame_ious2,frame_ious3=[],[],[]
    id_v=0

    for ious in ious_collect_detection_free_track:
        frame_ious1.extend(ious)
    for ious in ious_collect_proposed_timer:
        frame_ious2.extend(ious)
    for ious in ious_collect_proposed_notimer:
        frame_ious3.extend(ious)
    plt.figure()
    upperzs=0.4
    print("The number of frame iou<0.3:", np.count_nonzero(np.array(frame_ious1) < upperzs))
    print("The number of frame iou<0.3:", np.count_nonzero(np.array(frame_ious2) < upperzs),np.count_nonzero(np.array(frame_ious2) <upperzs)/np.count_nonzero(np.array(frame_ious1)< upperzs))
    print("The number of frame iou<0.3:", np.count_nonzero(np.array(frame_ious3) < upperzs), np.count_nonzero(np.array(frame_ious3) < upperzs)/np.count_nonzero(np.array(frame_ious1)< upperzs))
    bins = np.linspace(0, 1, 10)
    print("frame average",np.average(frame_ious1),np.average(frame_ious2),np.average(frame_ious3))
    plt.hist([frame_ious1,frame_ious2,frame_ious3], bins, label=['time refresher', 'detect by track','update timer'])
    plt.legend(loc='upper left')
    plt.savefig('figs/frame_dist_compare.pdf' )
    plt.show()

def draw_video_dis_collect(ious_collect_detection_free_track,ious_collect_proposed_timer):
    video_ious1,video_ious2=[],[]
    id_v=0

    for ious in ious_collect_detection_free_track:
        video_ious1.append(np.average(np.array(ious)))
    for ious in ious_collect_proposed_timer:
        video_ious2.append(np.average(np.array(ious)))
    print("video average", np.average(video_ious1), np.average(video_ious2))
    plt.figure()

    bins = np.linspace(0, 1, 10)

    plt.hist([video_ious1,video_ious2], bins, label=['time refresher', 'detect by track'])
    plt.legend(loc='upper left')
    plt.savefig('figs/video_dist_compare.pdf' )
    plt.show()
def draw_video_dis2d_collect(ious_collect_detection_free_track,ious_collect_proposed_timer):
    video_ious1,video_ious2=[],[]
    id_v=0

    for ious in ious_collect_detection_free_track:
        video_ious1.append([np.average(np.array(ious)),np.std(np.array(ious))])
    for ious in ious_collect_proposed_timer:
        video_ious2.append([np.average(np.array(ious)),np.std(np.array(ious))])

    plt.figure()

    plt.scatter([iou[0] for iou in video_ious1],[iou[1] for iou in video_ious1],label='time refresher',alpha=0.5)
    plt.scatter([iou[0] for iou in video_ious2],[iou[1] for iou in video_ious2],label='detect by track',alpha=0.5)

    plt.legend(loc='upper left')
    plt.savefig('figs/video_dist2d_compare.pdf' )
    plt.show()
def draw_video_dis(ious_collect,name='video_ious_dist.png'):
    video_ious = []
    for ious in ious_collect:
        video_ious.append(np.average(np.array(ious)))
    plt.figure()
    plt.hist(video_ious)
    plt.savefig('figs/'+name)
   # plt.show()
def draw_video_dis_2d(ious_collect,name='video_ious_dist2d.png'):
    mean_ious = []
    std_ious=[]
    for ious in ious_collect:
        mean_ious.append(np.average(np.array(ious)))
        std_ious.append(np.std(np.array(ious)))
    plt.figure()
    plt.scatter(mean_ious,std_ious)
    plt.savefig('figs/'+name )
   # plt.show()

def get_IoUs(areas_new,areas_old):
   # a, b = np.asarray(areas_old), np.asarray(areas_new)
    out=np.zeros((len(areas_new), len(areas_old)))
    for i, new in enumerate(areas_new):
        for j,old in enumerate(areas_old):
            out[i,j]=intersect_area(old,new)/(area(new)+area(old)-intersect_area(old,new))
    return out
def mullist2arr(data):
    out=[]
    for d in data:
        out.extend(d)
    return out

def explore_tracking_performance(compression_level):
    # ious_collect_detection_free_track,mis_times_collect_detection_free_track,win_times__collect_detection_free_track,track_props_collect_detection_free_track=collectDistances(compression_level,track_key='_features_all_detection_free_track.npy',detect_key='features_all.npy')
    # ious_collect_proposed_timer,mis_times_collect_proposed_timer,win_times_collect_proposed_timer,track_props_collect_proposed_timer=collectDistances(compression_level,track_key='_features_all_autoupdate_timer.npy',detect_key='features_all.npy')
    # np.save("data/detection_free_track.npy", [ious_collect_detection_free_track,mis_times_collect_detection_free_track,win_times__collect_detection_free_track,track_props_collect_detection_free_track])
    # np.save("data/proposed_scheme_timer.npy", [ious_collect_proposed_timer,mis_times_collect_proposed_timer,win_times_collect_proposed_timer,track_props_collect_proposed_timer])
    #
    # ious_collect_proposed_notimer,mis_times_collect_proposed_notimer,win_times_collect_proposed_notimer,track_props_proposed_notimer=collectDistances(compression_level,track_key='_features_all_track_by_detect_onlytrack.npy',detect_key='features_all.npy')
    # np.save("data/update_only.npy", [ious_collect_proposed_notimer,mis_times_collect_proposed_notimer,win_times_collect_proposed_notimer,track_props_proposed_notimer])
    # ious_collect_detection_free_timer,mis_times_collect_detection_free_timer,win_times_collect_detection_free_timer,track_props_detection_free_timer=collectDistances(compression_level,track_key='_features_all_time_refresher.npy',detect_key='features_all.npy')
    # np.save("data/detection_free_track_timer.npy", [ious_collect_detection_free_timer,mis_times_collect_detection_free_timer,win_times_collect_detection_free_timer,track_props_detection_free_timer])


    ious_collect_detection_free_track,mis_times_collect_detection_free_track,win_times__collect_detection_free_track,track_props_collect_detection_free_track=np.load("data/detection_free_track.npy", allow_pickle=True)
    ious_collect_proposed_timer,mis_times_collect_proposed_timer,win_times_collect_proposed_timer,track_props_collect_proposed_timer=np.load("data/proposed_scheme_timer.npy", allow_pickle=True)
    ious_collect_proposed_notimer, mis_times_collect_proposed_notimer, win_times_collect_proposed_notimer, track_props_proposed_notimer =np.load("data/update_only.npy", allow_pickle=True)
    ious_collect_detection_free_timer, mis_times_collect_detection_free_timer, win_times_collect_detection_free_timer, track_props_detection_free_timer=np.load("data/detection_free_track_timer.npy", allow_pickle=True)


    upperzs=0.3
    print("The number of frame detection-free tracking iou<0.3:", np.count_nonzero(np.array(mullist2arr(ious_collect_detection_free_track)) < upperzs))
    print("The number of frame proposed no timer iou<0.3:", np.count_nonzero(np.array(mullist2arr(ious_collect_proposed_notimer)) < upperzs),
      np.count_nonzero(np.array(mullist2arr(ious_collect_proposed_notimer)) < upperzs) / np.count_nonzero(np.array(mullist2arr(ious_collect_detection_free_track)) < upperzs))


    print("The number of frame detection-free tracking timer iou<0.3:",
      np.count_nonzero(np.array(mullist2arr(ious_collect_detection_free_timer)) < upperzs))

    print("The number of frame proposed timer iou<0.3:", np.count_nonzero(np.array(mullist2arr(ious_collect_proposed_timer)) < upperzs),
      np.count_nonzero(np.array(mullist2arr(ious_collect_proposed_timer)) < upperzs) / np.count_nonzero(np.array(mullist2arr(ious_collect_detection_free_timer)) < upperzs))

    # plt.figure(figsize=(10, 4.5))
    # bins = np.linspace(0, 1, 10)
    # plt.hist([mullist2arr(ious_collect_detection_free_timer), mullist2arr(ious_collect_proposed_timer),mullist2arr(ious_collect_detection_free_track), mullist2arr(ious_collect_proposed_notimer)], bins, label=['detection-free tracking', 'dead-tracking-n-detection','detection-free tracking-no', 'dead-tracking-n-detection-no'])
    # plt.xticks(fontsize=16)
    # plt.xlabel(r'$IOU$', fontsize=16)
    # plt.ylabel('count',fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.legend(fontsize=16)
    # plt.savefig('figs/iou_compare_timerasfs.pdf', bbox_inches='tight')
    #
    # plt.figure(figsize=(10, 4.5))
    # bins = np.linspace(0, 1, 10)
    # plt.hist([mullist2arr(ious_collect_detection_free_timer), mullist2arr(ious_collect_proposed_timer)], bins, label=['detection-free tracking', 'dead-tracking-n-detection'])
    # plt.xticks(fontsize=16)
    # plt.xlabel(r'$IOU$', fontsize=16)
    # plt.ylabel('count',fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.legend(fontsize=16)
    # plt.savefig('figs/iou_compare_timer.pdf', bbox_inches='tight')
    # plt.show()    # draw_video_dis_2d(ious_collect_detection_free_track,ious_collect_proposed_timer)
    fig=plt.figure(figsize=(10, 4.5))
    bins = np.linspace(0, 1, 10)
    plt.hist([mullist2arr(ious_collect_detection_free_track), mullist2arr(ious_collect_proposed_notimer)], bins, label=['Detection-free tracking',  'Deduced face detection'])
    plt.xticks(fontsize=16)
   # plt.text(-0.04,230000, r'$\times 10^5$', fontsize=16)
    plt.xlabel(r'$IoU$', fontsize=16)
    plt.ylabel('count',fontsize=16)
    ax1 = plt.gca()
    ax1.set_yticklabels(labels=[0, 0.5,1.0,1.5,2.0],
                        fontsize=16)
    #ax1.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.yticks(fontsize=16)
    plt.yscale('log')
    plt.legend(fontsize=16)
    # plt.axes([0.23,0.4,0.3,0.26])
    # plt.hist([mullist2arr(ious_collect_detection_free_track), mullist2arr(ious_collect_proposed_notimer)], np.linspace(0, 0.3, 10),
    #          label=['Detection-free tracking', 'Deduced face detection'])
    #ax1.set_yticklabels(labels=[0, r'$5\times10^4$', r'$1\times10^5$', r'$1.5\times10^5$', r'$2\times10^5$'], fontsize=16)

    # plt.xticks(np.linspace(0, 0.3, 3), fontsize=16)
    # plt.xlabel(r'$IoU$', fontsize=16)
    # plt.ylabel('count',fontsize=16)
    # plt.yticks(fontsize=16)
    plt.savefig('figs/iou_compare_no_timer.pdf', bbox_inches='tight')
    # plt.show()    # draw_video_dis_2d(ious_collect_detection_free_track,ious_collect_proposed_timer)
   # print(track_props_collect_detection_free_track)
    plt.figure(figsize=(10, 4.5))
    #bins = np.linspace(0, 1, 10)
    DR_difference_timer=[]
    DR_difference_notimer =[]

    times_timer=0
    times_notimer=0
    for i in range(len(track_props_detection_free_timer)):
        d=track_props_collect_proposed_timer[i]-track_props_detection_free_timer[i]
        if d>0:
            DR_difference_timer.append(d)
        d=track_props_proposed_notimer[i]-track_props_collect_detection_free_track[i]
        if d>0:
            DR_difference_notimer.append(d)

    print("DR difference",np.sum(np.array(DR_difference_timer)),np.sum(np.array(DR_difference_notimer)),DR_difference_timer,DR_difference_notimer)

    plt.hist([track_props_collect_detection_free_track,track_props_proposed_notimer], label=['Detection-free tracking', 'Deduced face detection'])
    plt.xticks(fontsize=16)
    plt.xlabel(r'$DR$', fontsize=16)
    plt.ylabel('count',fontsize=16)
    plt.yscale('log')
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig('figs/track_prob_compare.pdf', bbox_inches='tight')
    plt.show()    # draw_video_dis_2d(ious_collect_detection_free_track,ious_collect_proposed_timer)


def intersect_area(a,b):
    x1=max(a[0],b[0])
    x2=max(a[1],b[1])
    x3=min(a[2],b[2])
    x4=min(a[3],b[3])
    return area([x1,x2,x3,x4])


for compression_level in [0]:
    print("Compression level: ",compression_level)
   # rename_file(compression_level)
    explore_tracking_performance(compression_level)
