from scipy.signal import find_peaks
import json
import os
import cv2
import numpy as np
import dlib
from VideoPerson import *
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
import time
import pickle

face_rec_model_path = "C:/Users/Tangli/Downloads/codes/dlib_face_recognition_resnet_model_v1.dat"
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
detector = dlib.get_frontal_face_detector()
predictor_path = "C:/Users/Tangli/Downloads/codes/shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(predictor_path)

already_exists=False
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

    return r2
def detect_face(image,roi=None,margin=50):
    face_positions=[]
    if roi is None:
        dets = detector(image, 1)
        flag = True
        if len(dets) == 0:
            flag = False
            return flag, []
        for det in dets:
         #   print('face',det.left(),det.top(),det.right(),det.bottom())
            left,top,right,bottom=det.left(),det.top(),det.right(),det.bottom()
            face_positions.append([left,top,right,bottom])
          #  print('face_positions', face_positions)
    else:
        start_x, start_y, end_x, end_y=roi
    #    print("start_x, start_y, end_x, end_y",start_x, start_y, end_x, end_y)
        h,w,p=image.shape
        dets = detector(image[max(start_y-margin,0):min(end_y+margin,h), max(start_x-margin,0):min(end_x+margin,w)], 1)
       # dets = detector(image[:][100:], 1)#468 82 735 350   419 63 740 384
      #  print(max(start_x-margin,0),min(end_x+margin,w),max(start_y-margin,0),min(end_y+margin,h))
      #  print(image.shape,(image[start_y-margin:end_y+margin, start_x-margin:end_x+margin]).shape)
        flag = True
        if len(dets) == 0:
            flag = False
            return flag, []
        else:

            for det in dets:
                left, top, right, bottom = det.left(), det.top(), det.right(), det.bottom()
                print("f",left, top, right, bottom)
                print("detected", left-margin+start_x, top-margin+start_y, right-margin+start_x, bottom-margin+start_y)
                face_positions.append([left-margin+start_x, top-margin+start_y, right-margin+start_x, bottom-margin+start_y])
               # face_positions.append([left, top+100 , right , bottom+100 ])

    return flag,face_positions
def read_mvs_data(mvs_path):
    with open(mvs_path) as input_file:
       mvs_data = json.load(input_file)
    index=0

    pos_x=[]
    pos_y=[]
    while index<len(mvs_data):
        item=mvs_data[index]

        pos_x.append(item['dst_x'])
        pos_y.append(item['dst_y'])
        index+=1
    return [pos_x,pos_y]
def moving_object_num(mvs_data,shape):
    mvs_count=0
    threshold=100
    pos_x,pos_y=mvs_data

    bins_x=np.arange(0,shape[0]+128,128)
    hists_x, bins_x =np.histogram(pos_x,bins_x)
    peaks_x, _ = find_peaks(hists_x,distance=5,height=10)#低于height的不考虑


    return len(peaks_x)#max(len(peaks_x),len(peaks_y))
def if_detect_for_whole(video_info,cur_person_num,frame_count,refresh_timer):
    mvs_data, video_size, video_fps=video_info
    detect_flag=True
    if frame_count != 0:
        object_num = moving_object_num(mvs_data, video_size)
        if object_num > cur_person_num or refresh_timer == 0:
            detect_flag = True
        else:
            detect_flag = False
    return detect_flag
def draw_landmarks(frame,face_id,face_area,landmarks):
    left,top,right,bottom=face_area
   # print("left,top,right,bottom",left,top,right,bottom)
    cv2.rectangle(frame, (left,top),(right,bottom), (0, 255, 0), 2)
  #  print("image",frame[top:bottom, left:right])
    cv2.putText(frame,'person'+str(face_id),(left,top),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        cv2.circle(frame, pos, radius=0, color=(0, 0, 255), thickness=2)
def compare_distance(p1,p2):
    face_distance=[]
    lip_distance=[]
    print('range',min(p1[0].frame_pos,p2[0].frame_pos),max(p1[-1].frame_pos,p2[-1].frame_pos))
    for i in range(min(p1[0].frame_pos,p2[0].frame_pos),max(p1[-1].frame_pos,p2[-1].frame_pos)):
        print("frame_pos",p1[i].frame_pos,p2[i].frame_pos)
        if p1[i].frame_pos==p2[i].frame_pos:
            print("distance",_euclidean_distance([p1[i].face_descriptor],[p2[i].face_descriptor]))
            face_distance.append(_euclidean_distance([p1[i].face_descriptor],[p2[i].face_descriptor])[0,0])
    return face_distance





def get_features(frame,frame_pos,face_position):
    face=dlib.rectangle(face_position[0], face_position[1], face_position[2], face_position[3])
    shape = shape_predictor(frame, face)
    face_pos = [face.left(), face.top(), face.right(), face.bottom()]
    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    positions_lip_arr = []
    for idx, point in enumerate(landmarks[48:68]):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        positions_lip_arr.append(pos)

    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
    face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
    person=FaceFeature(face_descriptor, positions_lip_arr, face_pos, frame_pos)
    return person,landmarks


def merge_video(original_video,forged_video,merged_video,insert_len=100):
    cap_ori = cv2.VideoCapture(original_video)
    frame_num_ori = int(cap_ori.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_forged = cv2.VideoCapture(forged_video)
    frame_num_forged = int(cap_forged.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_num_ori, frame_num_forged)
    step = 25
    cut_start = 102#np.random.randint(0, min(frame_num_ori, frame_num_forged) - insert_len, size=1)[0]

    video_fps = cap_ori.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap_ori.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap_ori.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(merged_video, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, video_size, True)
    frame_count=0

    while True:
        success, frame_ori = cap_ori.read()
        if not success:
            break
        success, frame_forged = cap_forged.read()
        if not success:
            break
        if frame_count>=cut_start and frame_count<cut_start+insert_len:
            out.write(frame_forged)
        else:
            out.write(frame_ori)
        frame_count+=1
    out.release()
    cap_ori.release()
    cap_forged.release()
    print("cut_start",cut_start)
    return [int(cut_start/step),math.ceil((cut_start+insert_len)/step)],cut_start

def getFaceArea(image):

    dets = detector(image, 1)
    flag = True
    if len(dets) == 0:
        flag = False
        return flag, []
    # print("dets",self.img_rgb)
    # print("Number of faces detected: {}".format(len(dets)))
    return flag,dets
def genFeatures(video_path):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    detect_flag=True
    refresh_timer=video_fps

    trackers=[]
    frame_count=0
    video_data=[]

    while True:
        success, frame = cap.read()
        # print(real_frame)
        if not success:
            break

        persons=[]
        if detect_flag:
            trackers = []
            b, g, r = cv2.split(frame)
            img_rgb = cv2.merge([r, g, b])
            flag, faces = getFaceArea(img_rgb)  # 提取128维向量，是dlib.vector类的对象
            for index, face in enumerate(faces):
                shape = shape_predictor(frame, face)
                face_pos = [face.left(), face.top(), face.right(), face.bottom()]
                landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])

                positions_lip_arr = []
                for idx, point in enumerate(landmarks[48:68]):
                    # 68点的坐标
                    pos = (point[0, 0], point[0, 1])
                    positions_lip_arr.append(pos)
                face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
                persons.append([face_descriptor, face_pos, positions_lip_arr])
                tracker = dlib.correlation_tracker()
                # Start a track on face detected on first frame.
                rect = dlib.rectangle(face.left(), face.top(), face.right(), face.bottom())
                tracker.start_track(frame, rect)
                trackers.append(tracker)

        video_data.append([frame_count, persons])
        frame_count += 1

    cap.release()


    return video_data
def euclidean_distance(a, b):
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
def get_lip_attributes(lip_landmarks, lip_size):

    d1,d2,d3,d4=[],[],[],[]
    print("lip_landmarks",lip_landmarks[6][0])
    # print("lip_landmarks22",lip_landmarks[6][0])
    print("lip_size",lip_size)
    d1.append((lip_landmarks[6][0]-lip_landmarks[0][0])/lip_size[0])#width
    d1.append((lip_landmarks[9][1] - lip_landmarks[3][1])/lip_size[1])#height
   # print(d1)
    for i in range(1,6):
        d2.append((lip_landmarks[12-i][1] - lip_landmarks[i][1])/lip_size[1])
    center_point=[((lip_landmarks[6][0]+lip_landmarks[0][0])/2),(lip_landmarks[9][1] + lip_landmarks[3][1])/2]
  #  print("d2")
    # for i in range(12):
    #     d3.append(euclidean_distance([[lip_landmarks[i][0]/lip_size[0],lip_landmarks[i][1]/lip_size[1]]],[[center_point[0]/lip_size[0],center_point[1]/lip_size[1]]])[0,0])
    # print("d3")
    # for i in range(12,20):
    #     d4.append(euclidean_distance([[lip_landmarks[i][0] / lip_size[0], lip_landmarks[i][1] / lip_size[1]]],
    #                                  [[center_point[0] / lip_size[0], center_point[1] / lip_size[1]]])[0, 0])
    # print("d1,d2,d3,d4",d1,d2,d3,d4)
    return d1,d2,d3,d4
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
               # print(cur_data[j][1][0][feature_id])
                lip_attributes1 = get_lip_attributes(ref_data[i-1][1][0][feature_id], lip_size)
                lip_attributes2 = get_lip_attributes(cur_data[j][1][0][feature_id], lip_size)
               # print("last_i:{}, j:{}, next_i:{}".format( ref_data[i-1][0], cur_data[j][0],ref_data[i][0]))
                print([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])
                dis = euclidean_distance([lip_attributes1[attr_i]], [lip_attributes2[attr_i]])[0, 0]
                print("dis:",dis)
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
def groupdata(data):
    window_size,step_size=50,25
    out=[]
    for i in range(int(len(data)/step_size)):
        out.append(np.mean(np.array(data[i*step_size:min(len(data),i*step_size+window_size)])))
    print("average",out)
    return out
# original_video='D:/TangLi/datasets/FaceForensics++/original_sequences/youtube/c23/videos/000.mp4'
# forged_video='result_voice.mp4'
# merged_video='lip_syncing_merged.mp4'
# # tamper_range,cut_start=merge_video(original_video,forged_video,merged_video)
# ref_data=genFeatures(original_video)
# np.save("partial_lip_syncing_ref_data.npy",ref_data)
# fake_data=genFeatures(merged_video)
# np.save("partial_lip_syncing_fake_data.npy",fake_data)
ref_data=np.load("partial_lip_syncing_ref_data.npy",allow_pickle=True)
fake_data=np.load("partial_lip_syncing_fake_data.npy",allow_pickle=True)
distance_data=getFeatureDistances(ref_data,fake_data,0,get_lip_size(ref_data))
print(distance_data)
np.save('distance_merged_lip_synicing_attribute_a.npy',[distance_data,[4, 9], 102])

distance_data,tamper_range,cut_start=np.load('distance_merged_lip_synicing_attribute_a.npy',allow_pickle=True)
print(tamper_range,cut_start)
#print([[i,distance_data[i][2]] for i in range(len(distance_data))])
#forged_segment_mean,forged_segment_model=authen_comparision(distance_data)
#print(tamper_range,forged_segment_mean,forged_segment_model)
segment_data=groupdata(distance_data)
#肉眼观察的话 篡改的其实是125到241
step=25
plt.figure(figsize=(9,4))
plt.plot([distance_data[i] for i in range(len(distance_data))],label='distance')
#plt.plot([d[0]*step+step for d in segment_data],[d[1][0] for d in segment_data],linestyle='dotted')
plt.plot([i*step+step for i in range(len(segment_data))],[d for d in segment_data],'ro',linestyle='dotted',markerfacecolor='white',label='mean value')
#plt.plot([index*step+step for index in forged_segment_model],[segment_data[index][1][0] for index in forged_segment_model],'rx')
plt.plot([index*step+step for index in [4,5,6,7]],[segment_data[index] for index in [4,5,6,7]],'rx')
plt.ylabel('Distance',fontsize=16)
plt.xlabel('Index',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.savefig('partial_lip_syncing_authen_attribute_a.pdf',bbox_inches='tight')
plt.show()