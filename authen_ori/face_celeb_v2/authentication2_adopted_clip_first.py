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
def merge_videoclip(crf_distances,fake_distances,insert_len=100):
    out=[]
  #  print(fake_distances,crf_distances)
    #print(len(crf_distances), len(fake_distances),insert_len,min(len(crf_distances), len(fake_distances)) - insert_len)
    cut_start = 0#np.random.randint(0, max(0,min(len(crf_distances), len(fake_distances)) - insert_len)+1, size=1)[0]
    frame_count=0
    while frame_count<len(crf_distances) and frame_count<len(fake_distances):

        if frame_count>=cut_start and frame_count<cut_start+insert_len:
            out.append(fake_distances[frame_count])
        else:
            out.append(crf_distances[frame_count])
        frame_count+=1
    return out,[cut_start,min(cut_start+insert_len,frame_count)]
def create_videoclips(crf_distances_set,fake_distances_set,insert_len):
    out=[]
    clip_ranges,frame_ranges=[],[]
   # print(len(crf_distances_set),len(fake_distances_set))
    for i in range(len(crf_distances_set)):
       # print(len(fake_distances_set[i]))
        for j in range(len(fake_distances_set[i])):
           # insert_len = np.random.randint(30, max(int(len(crf_distances_set[i])/2),31), size=1)[0]
            if np.max(crf_distances_set[i])>0.5:
               print("error a",np.max(crf_distances_set[i]))
            elif np.min(fake_distances_set[i][j])<0.5:
               print("error f",np.min(fake_distances_set[i][j]))
            dis,frame_range=merge_videoclip(crf_distances_set[i], fake_distances_set[i][j], insert_len=insert_len)
            out.append(dis)

            frame_ranges.append(frame_range)
    return out,frame_ranges

def reorganize(crf_distances_set,fake_distances_set):
    ori_pathes, fake_paths, crf_paths,mul_people_video = loadFeatureFiles(compression_level, '_features_all.npy')

    new_crf_distances_set, new_fake_distances_set = [], []
    crf_id, fake_id = 0, 0
   # print(len(ori_pathes))
    for i in range(len(ori_pathes)):
        ori_path=ori_pathes[i]
        crf_path=crf_paths[i]

        if ori_path in mul_people_video or crf_path in mul_people_video:
            continue
      #  print(len(crf_distances_set))
        if crf_id>=len(crf_distances_set):
            break
        new_crf_distances_set.append(crf_distances_set[crf_id])
        crf_id+=1
        temp = []
        for fake_path in fake_paths[i]:
            #   print("here stack")
            if fake_path in mul_people_video:
                continue
            #  print("processing",fake_path)
            # print(fake_id,len(fake_distances_set))
            if fake_id >= len(fake_distances_set):
                break
            temp.append(fake_distances_set[fake_id])
            fake_id += 1
        new_fake_distances_set.append(temp)

   # print(fake_distances)
    return np.array(new_crf_distances_set), np.array(new_fake_distances_set)
def get_segment_statistics(data,window,step):
    output=[]
   # print("segment num",int(len(data)/window))

    for i in range(int(len(data)/step)+1):
        start=i*step
        end=min(i*step+window,len(data))
       # print(start,end,len(data))
        if start>=end:
            break
        output.append(np.mean(data[start:end]))
    # if end<len(data):
    #     output.append(np.mean(data[end:]))
    return output
def find_forged_segments(data,threshold):
    #print(threshold,data)
    forged_segment=[]
    for i in range(len(data)):
        if data[i]>threshold:
            forged_segment.append(i)
    return forged_segment
def get_samples_segments(judged,ref):
    tp, fp, tn, fn = 0, 0, 0, 0
    for index in judged:
        if index<=ref[1] and index>=ref[0]:
            tp+=1
        else:
            fp+=1
    fn=ref[1]-ref[0]+1-tp
    return tp,fp,fn



def getAuthenPerformance(re_crf_distances_clips,re_fake_distances_clips,frame_ranges,thd,window=50,step=25):
    tp, fp, fn=0,0,0
    tp_v, fp_v, fn_v = 0, 0, 0
    scores=[]
    labels=[]
    scores_v=[]
    labels_v=[]
    print(len(re_crf_distances_clips),len(re_fake_distances_clips))
    for i in range(len(re_fake_distances_clips)):
        clip_scores = get_segment_statistics(re_fake_distances_clips[i],window,step)
        scores.extend([[s] for s in clip_scores])

        scores_v.append(clip_scores[0])
        labels_v.append([1])
     #   print("video length:{}, window:{},group number:{} , score number:{}".format(len(re_fake_distances_clips[i]),window,int(len(re_fake_distances_clips[i])/window),len(clip_scores)))
        # plt.figure()
        # plt.plot(re_fake_distances_clips[i])
        # print("range",frame_ranges[i])
        # print(clip_scores)
        # plt.show()
        temp=[]

        clip_range=[int(frame_ranges[i][0]/step),int(frame_ranges[i][1]/step)]
        for j in range(len(clip_scores)):
            if j>=clip_range[0] and j<=clip_range[1]:
                temp.append([1])
            else:
                temp.append([-1])
        labels.extend(temp)
        predicted_forged_segment = find_forged_segments(clip_scores, thd)
        tp_c, fp_c, fn_c = get_samples_segments(predicted_forged_segment, clip_range)
        if predicted_forged_segment !=[]:
            tp_v+=1
        else:
            fn_v+=1
        tp+=tp_c
        fp+=fp_c
        fn+=fn_c
    for i in range(len(re_crf_distances_clips)):
        clip_scores = get_segment_statistics(re_crf_distances_clips[i],window,step)
        scores.extend([[s] for s in clip_scores])
        labels.extend([[-1] for s in clip_scores])
        scores_v.append(clip_scores[0])
        labels_v.append([-1])
        predicted_forged_segment = find_forged_segments(clip_scores, thd)
        if predicted_forged_segment!=[]:
            fp_v+=1
        fp+=len(predicted_forged_segment)

    #print("scores",scores_v)

    roc_auc = roc_auc_score(np.array(labels).ravel(), np.array(scores).ravel())
    recall,precision,acc=tp/(tp+fn),tp/(fp+tp),(len(scores)-fp-fn)/len(scores)


    roc_auc_v = roc_auc_score(np.array(labels_v).ravel(), np.array(scores_v).ravel())
    recall_v,precision_v,acc_v=tp_v/(tp_v+fn_v),tp_v/(fp_v+tp_v),(len(scores_v)-fp_v-fn_v)/(len(scores_v))
    print("Final Result for mean clip: ROC:{}, Recall: {}, Precision:{}, Accuracy:{} ".format(roc_auc,recall,precision,acc))
    print("Final Result for mean videolevel: ROC:{}, Recall: {}, Precision:{}, Accuracy:{} ".format(roc_auc_v, recall_v, precision_v,acc_v))
   # print("Final Result for model: Recall: {}, Precision:{} ".format(tp_f / (tp_f + fn_f), tp_f / (fp_f + tp_f)))
    return [roc_auc,recall,precision,acc],[roc_auc_v, recall_v, precision_v,acc_v]



def find_best_range(ths,metrics,bench=0):
    precision=[metrics[i][0] for i in range(len(metrics))]
    recall=[metrics[i][1] for i in range(len(metrics))]
    acc=[metrics[i][2] for i in range(len(metrics))]
    th_range=[]
    best_range=[]
    if bench==0:
        # take acc as the main: find the 0.95%best result
        limits = 0.999
        best_idx = np.argmax(acc)
        for i in range(len(ths)):
            if acc[i] >= acc[best_idx] * limits:
                th_range.append(ths[i])
                if acc[i] == acc[best_idx]:
                 #   print("sd",acc[i],acc[best_idx])
                    best_range.append(ths[i])
    elif bench==1:
        # take 0.95% acc范围去寻找最好的precision
        limits = 0.999
        best_idx = np.argmax(acc)
        in_recall=[]
        for i in range(len(ths)):
            if acc[i] >= acc[best_idx] * limits:
                th_range.append(ths[i])
                in_recall.append(recall[i])
        best_pre_idx = np.argmax(in_recall)
        for i in range(len(in_recall)):
            if in_recall[i]>=in_recall[best_pre_idx]* limits:
                best_range.append(th_range[i])

    return th_range,best_range
def clear_data(distances):
    out=[]
    for d in distances:
        if d==[]:
            continue
        out.append(d)
    return out
def explore_clip_detection_vs_window(compression_level,insert_len):
    #windows=[50,100,150,200,250]
    windows=[10,20,30,40,50,60,70,80,90,100,110,120]
    ths = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    th_data = []
    ylims = [(0.999, 1), (0.99, 1.), (.93, 1)]
    metrics_set_levels = np.load('data/face_3metrics_vs_comp&ts.npy', allow_pickle=True)
    aucs_v, aucs_clip = [], []

    for w_i in range(len(windows)):
        if windows[w_i]==90:
            th, crf_distances, fake_distances, saves = np.load(
                'data/celebv2_comp' + str(compression_level) + '_threshold' + str(0) + '_distances_curall.npy',
                allow_pickle=True)
            print("t_s=", th, saves)
            re_crf_distances, re_fake_distances = reorganize(clear_data(crf_distances), clear_data(fake_distances))
            re_fake_distances_clips, frame_ranges = create_videoclips(re_crf_distances, re_fake_distances, insert_len)
            np.save('data/celebv2_comp' + str(compression_level) + '_insert_len' + str(insert_len) + '_video_clips.npy',
                    [re_crf_distances, re_fake_distances_clips, frame_ranges])
        else:
            re_crf_distances, re_fake_distances_clips, frame_ranges = np.load(
                'data/celebv2_comp' + str(compression_level) + '_insert_len' + str(insert_len) + '_video_clips.npy',
                allow_pickle=True)


        re_crf_distances, re_fake_distances_clips=np.array(re_crf_distances), np.array(re_fake_distances_clips)
       # print(len(re_fake_distances),len(re_fake_distances_clips))
        best_range, best = find_best_range(ths, metrics_set_levels[compression_level][0], 1)
        thd = np.min(best)
        print("window size:{}, decision threshold;{}".format(windows[w_i],thd))
        result_clip, result_video = getAuthenPerformance(
        re_crf_distances,re_fake_distances_clips[np.random.choice(re_fake_distances_clips.shape[0], re_crf_distances.shape[0], replace=False)], frame_ranges,
        thd, window=windows[w_i], step=int(windows[w_i] / 2))

  #      result_clip,result_video= getAuthenPerformance(re_crf_distances[np.random.choice(re_crf_distances.shape[0], 500, replace=False)],re_fake_distances_clips[np.random.choice(re_fake_distances_clips.shape[0], 500, replace=False)],frame_ranges,thd,window=windows[w_i],step=int(windows[w_i]/2))
        th_data.append([result_clip,result_video])
        aucs_v.append(result_video[0])
        aucs_clip.append(result_clip[0])
    return windows, aucs_v, aucs_clip

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
    plt.figure()
    plt.scatter([d[0] for d in crf_score],[d[1] for d in crf_score],label='Authentic')
    plt.scatter([d[0] for d in fake_score], [d[1] for d in fake_score], label='Lip-Syncing')

    plt.xticks(fontsize=20)
    plt.xlabel(r'$\mu$',fontsize=20)
    plt.ylabel(r'$\sigma$',fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(name)
  #  plt.show()
repeat_times=5
windows=[10,20,30,40,50,60,70,80,90,100,110,120]
ylims = [[(0.9943, 1), (0.9965, 1), (.997, 1)],[(0.931, 1), (0.9965, 1), (.997, 1)],[(0.67, 1), (0.91, 1), (.979, 1)]]
for compression_level in [0,1,2]:
    print("Compression level: ", compression_level)

    fig = plt.figure(figsize=(9,6))
    colors=['red','green','blue']
    markers=['o','^','s']
    insert_lens=[30,60,90]
    for len_i in range(len(insert_lens)):
        # aucs_vset, aucs_clipset = [], []
        # for i in range(repeat_times):
        #     print("Teeppppppppppppppppppppppppp")
        #     print(insert_lens[len_i])
        #     windows, aucs_v, aucs_clip = explore_clip_detection_vs_window(compression_level,insert_lens[len_i])
        #     aucs_vset.append(aucs_v)
        #     aucs_clipset.append(aucs_clip)
        # aucs_v_ave = np.mean(np.array(aucs_vset), axis=0)
        # aucs_clip_ave = np.mean(np.array(aucs_clipset), axis=0)
        # print("aucs", aucs_vset)
        # print("auc_ave", aucs_v_ave)
        # np.save("data/face_partial_detection_comp" + str(compression_level) + "_insert_len"+str(insert_lens[len_i])+"_temp.npy", [aucs_v_ave, aucs_clip_ave])
        aucss = np.load("data/face_partial_detection_comp" + str(compression_level) + "_insert_len"+str(insert_lens[len_i])+"_temp.npy", allow_pickle=True)
        aucs_v_ave, aucs_clip_ave = aucss
   #     aucss = np.load("data/face_partial_detection_comp" + str(compression_level) + "_insert_len"+str(insert_lens[len_i])+".npy", allow_pickle=True)
    #    aucs_v_ave, aucs_clip_ave = aucss
        ax = plt.subplot(3, 1, len_i + 1)
        plt.plot(windows, aucs_v_ave, color=colors[len_i], marker=markers[len_i], linestyle='solid',
                 label='$D_f=$' + str(insert_lens[len_i]))
        print("aucs_v_ave",aucs_v_ave)
        # print("saving",saving)
        plt.grid()
        plt.ylabel("AUC", fontsize=20)
        fff = plt.gca()
        plt.yticks(fontsize=20)
        plt.xlabel("$D_w$", fontsize=20)
        plt.ylim(ylims[compression_level][len_i])
        plt.legend(fontsize=20)

    #  fff.axes.get_xaxis().set_visible(False)
    # plt.xticks([])
    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0, hspace=0.05)  # 调整子图间距
    plt.xticks(fontsize=20)

    plt.savefig("figs/face_partial_fixed_len_detection_comp" + str(compression_level) + "_first_group.pdf")
    plt.show()
