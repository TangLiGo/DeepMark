
import os
import cv2
import numpy as np

from PIL import Image
import dlib
import json
from scipy.signal import find_peaks
from utilsghf import *

# grab the appropriate object tracker using our dictionary of
def match_iou(cur_faces,stored_faces,metric='euclidean'):
    if metric == "euclidean":
       metric = _euclidean_distance
    elif metric == "cosine":
        metric = _cosine_distance
    face_ious = get_IoUs([facetrack.features[0].face_area for facetrack in cur_faces], [facetrack.features[0].face_area for facetrack in stored_faces])
    sorted_iou_indices = np.argsort(-face_ious, axis=1)

    matched_id=[sorted_iou_indices[id][0] for id in range(len(cur_faces))]
    return matched_id



def get_mvs_info(mvs_path):
    with open(mvs_path) as input_file:
        mvs_data = json.load(input_file)
    pos_area_x = []
    pos_area_y = []

    index = 0
    while index < len(mvs_data):
        item = mvs_data[index]
        pos_area_x.append(item['dst_x'])
        pos_area_y.append(item['dst_y'])

    return pos_area_x, pos_area_y
def mvs_distribution_peak(mvs_data,shape):
    index=0

    pos_x=[]
    pos_y=[]
    while index<len(mvs_data):
        item=mvs_data[index]

        pos_x.append(item['dst_x'])
        pos_y.append(item['dst_y'])
        index+=1

    bins_x=np.arange(0,shape[0],64)
    hists_x, bins_x =np.histogram(pos_x,bins_x)
    peaks_x, _ = find_peaks(hists_x,distance=5,height=10)#低于height的不考虑

    return len(peaks_x)#max(len(peaks_x),len(peaks_y))
# def get_area_mvs_strength(mvs_data,face_pos,margin=0):
#     mvs_x, mvs_y=mvs_data
#     index=0
#    # print("face_pos",face_pos)
#     left, top, right, bottom = face_pos
#     pos_area_x=[]
#     pos_area_y=[]
#     while index<len(mvs_x):
#
#         if mvs_y[index]<bottom+margin and mvs_y[index]>top-margin and mvs_x[index]<right+margin and mvs_x[index]>left-margin:
#             pos_area_x.append(mvs_x[index])
#             pos_area_y.append(mvs_y[index])
#         index+=1
#     return len(pos_area_x)
def _euclidean_distance(a, b):
    # 用于计算成对的平方距离
    # a NxM 代表N个对象，每个对象有M个数值作为embedding进行比较
    # b LxM 代表L个对象，每个对象有M个数值作为embedding进行比较
    # 返回的是NxL的矩阵，比如dist[i][j]代表a[i]和b[j]之间的平方和距离
    # 实现见：https://blog.csdn.net/frankzd/article/details/80251042
    a, b = np.asarray(a), np.asarray(b)  # 拷贝一份数据
    if len(a) == 0 or len(b) == 0:
       # print("ec", len(np.zeros((len(a), len(b)))))
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(
        b).sum(axis=1)  # 求每个embedding的平方和
    # sum(N) + sum(L) -2 x [NxM]x[MxL] = [NxL]
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    #返回的矩阵 每一行i对应的是a[i] 与各个b[j] j=1-m 的距离

    return np.sqrt(r2)
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

class FaceFeature:
    def __init__(self, face_descriptor,lip_descriptor,face_area,frame_pos):
        # max age是一个存活期限，默认为70帧,在
        self.face_descriptor = face_descriptor
        self.lip_descriptor = lip_descriptor
        self.face_area = face_area
        self.frame_pos = frame_pos
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
class FaceTrack:
    # 一个轨迹的信息，包含(x,y,a,h) & v
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    """

    def __init__(self, face_id, n_init, max_age,frame_pos,
                 feature=None):
        # max age是一个存活期限，默认为70帧,在
        self._mode='detection'
        self.face_id = face_id
        self.hits = 1
        # hits和n_init进行比较
        # hits每次update的时候进行一次更新（只有match的时候才进行update）
        # hits代表匹配上了多少次，匹配次数超过n_init就会设置为confirmed状态
        self.age = 1 # 没有用到，和time_since_update功能重复
        self.time_since_update = 0
        # 每次调用predict函数的时候就会+1
        # 每次调用update函数的时候就会设置为0
        self.start=frame_pos
        self.end=frame_pos
        self.state = "Tentative"
        self.features = []
        # 每个track对应多个features, 每次更新都将最新的feature添加到列表中
        if feature is not None:
            self.features.append(feature)
      #  self._min_size=min_size
        self._n_init = n_init  # 如果连续n_init帧都没有出现失配，设置为deleted状态
        self._max_age = max_age  # 上限
def get_IoUs(areas_new,areas_old):
   # a, b = np.asarray(areas_old), np.asarray(areas_new)
    out=np.zeros((len(areas_new), len(areas_old)))
    for i, new in enumerate(areas_new):
        for j,old in enumerate(areas_old):
            out[i,j]=intersect_area(old,new)/(area(new)+area(old)-intersect_area(old,new))
    return out
class FaceTracker:
    # 是一个多目标tracker，保存了很多个track轨迹
    # 负责调用卡尔曼滤波来预测track的新状态+进行匹配工作+初始化第一帧
    # Tracker调用update或predict的时候，其中的每个track也会各自调用自己的update或predict
    """
    This is the multi-target tracker.
    """

    def __init__(self, metric="euclidean", iou_threshold=0.4, face_threshold=0.6,mvs_threshold=1,max_age=10, n_init=10,):
        # 调用的时候，后边的参数全部是默认的
        if metric == "euclidean":
            # 使用最近邻欧氏距离
            self.metric = _euclidean_distance
        elif metric == "cosine":
            # 使用最近邻余弦距离
            self.metric = _cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        # metric是一个类，用于计算距离(余弦距离或马氏距离)
        self.iou_threshold = iou_threshold
        # 最大iou，iou匹配的时候使用
        self.max_age = max_age
        # 直接指定级联匹配的cascade_depth参数
        self.n_init = n_init
        # n_init代表需要n_init次数的update才会将track状态设置为confirmed
        self.tracks = [] # 保存一系列轨迹
        self._next_id = 0 # 下一个分配的轨迹id
        self.face_threshold=face_threshold

        self.mvs_threshold=mvs_threshold

    def _print(self,face_id=-1,print_all=True):
        #print("PPPPPPPPP",len(self.tracks))
        if print_all:
            for face_id in range(len(self.tracks)):
                track=self.tracks[face_id]
                if track.state=='Confirmed':
                    print("The face info for person {person_id} is: Face area :{face_area}; Face duration: {start} - {end};".format(person_id=face_id,face_area=track.features[-1].face_area,start=track.start,end=track.end))
        else:
            track = self.tracks[face_id]
            print(
                "The face info for person {person_id} is: Face area :{face_area}; Face duration: {start} - {end}".format(
                    person_id=face_id, face_area=track.features[-1].face_area, start=track.start, end=track.end))


    def _match(self,faces,frame_pos,mvs_data):
        corr_ids=[]
        if frame_pos==0:
            for i in range(len(faces)):
                cur_id=self._creat_newtrack(faces[i], frame_pos)
                corr_ids.append(cur_id)
            return corr_ids
        targets = [t.features[-1] for t in self.tracks]
        face_ious=get_IoUs([face.face_area for face in faces],[f.face_area for f in targets])
        sorted_iou_indices=np.argsort(-face_ious,axis=1)
        feature_distances = self.metric([face.face_descriptor for face in faces], [f.face_descriptor for f in targets])
        sorted_feature_indices = np.argsort(feature_distances, axis=1)

        for i in range(len(faces)):
            iou_is_matched=False
            feature_is_matched=False
           # needs_redetection=True
            possible_new_face=False
            fj=0
           # print("begin",fj)
            loss_face_ids=[]
            similar_face_num=0
            # 判断在区域有交集的情况下，是否有同一人

            while fj < len(sorted_feature_indices[i]) and feature_distances[
                i, sorted_feature_indices[i, fj]] < self.face_threshold:
                feature_is_matched = True
                similar_face_num+=1
             #   print("here1")
                if face_ious[i, sorted_iou_indices[i, fj]] >self.iou_threshold:
                   # needs_redetection = False
                    iou_is_matched = True
                  #  print("here2")
                    break
                fj += 1
            # if fj==len(sorted_iou_indices[i]):
            #     print("no match")
            # else:
            #     print("match face distance", frame_pos, i, sorted_feature_indices[i, fj],feature_distances[i] , )
           # print("d", feature_distances[i], sorted_feature_indices[i])
          #  print('d-iou:',face_ious[i],sorted_iou_indices[i])
            if not feature_is_matched and not iou_is_matched:
                possible_new_face = True
            else:
                if feature_is_matched and iou_is_matched:
                    matched_id = sorted_feature_indices[i, fj]


                elif feature_is_matched and not iou_is_matched:

                    # should be scene change=================actually should also consider that there are two detected faces have the same face feature--static image
                    possible_scene_change = True
                    matched_id = sorted_feature_indices[i, fj - 1]
                if self.tracks[matched_id].end==frame_pos:
                    possible_new_face = True
                else:
                    self._update(matched_id, faces[i], frame_pos)



            if possible_new_face:
                matched_id = self._creat_newtrack(faces[i], frame_pos)

            #     if (frame_pos == 0 or get_area_mvs_strength(mvs_data, faces[i].face_area) > self.mvs_threshold):
            #         matched_id = self._creat_newtrack(faces[i], frame_pos)
            #     else:
            #         continue
            # print("frame pos:{}, id:{} mvslength{}".format(frame_pos, matched_id,
            #                                                get_area_mvs_strength(mvs_data, faces[i].face_area)))
            corr_ids.append(matched_id)
        return corr_ids
    def _creat_newtrack(self,faceinfo,frame_pos):
        new_facetrack = FaceTrack(self._next_id, self.n_init, self.max_age, frame_pos,
                                  feature=faceinfo)
        # print("frame pos mvs st",frame_pos,get_area_mvs_strength(mvs_data, detections[i].face_area))
        self.tracks.append(new_facetrack)
        self._next_id += 1
        return self._next_id - 1
    def _update(self,track_id,face_new_info,frame_pos):
        matched_face = self.tracks[track_id]
        success_flag=True
        if matched_face.state == 'FalseDetected':
            matched_face._n_init = self.n_init
            matched_face.start = frame_pos
            return success_flag
        face_distance=self.metric([face_new_info.face_descriptor], [matched_face.features[-1].face_descriptor])[0]
      #  print("update face distance",frame_pos,face_distance,)
        if face_distance<self.face_threshold:
            matched_face.features.append(face_new_info)
            matched_face.end = frame_pos
            return success_flag
        else:
            return not success_flag
    def _check_state(self,frame_pos):
        for t_id in range(self._next_id):
            t = self.tracks[t_id]
            if t.end != frame_pos:
                if t.state == 'Tentative':
                    t.state = 'FalseDetected'
                elif t.state == 'Confirmed':
                    #process the missing detected face
                    if t._max_age <= 0:
                        # already disappeared
                        t.state = 'Closed'
                    else:
                        # not sure if has disappeared
                        t._max_age -= 1
            else:
                #for detected faces

                if t.state == 'Tentative':
                    # This is a doubted face
                    if t._n_init <= 0:
                        t.state = 'Confirmed'
                    else:
                        # detect the newly
                        t._n_init -= 1

        return





                

