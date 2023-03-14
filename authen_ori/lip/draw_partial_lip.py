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

def genFeatures(video_path):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('id0_0008write.avi', cv2.VideoWriter_fourcc(*'XVID'), video_fps, video_size, True)
    detect_flag=True
    refresh_timer=video_fps
    persons=FaceTracker()
    trackers=[]
    frame_count=0
    video_data=[]
    person_num=0
    while True:
        success, frame = cap.read()
        if not success:
            break
        filename, t = os.path.splitext(video_path)
     #   mvs_path = filename + "/" + str(frame_count + 1) + "_mvs.json"
        mvs_data = []
        detect_flag=True#if_detect_for_whole([mvs_data, video_size, video_fps],len(trackers),frame_count,refresh_timer)

        if detect_flag:
            refresh_timer=video_fps
            tracked_face_ids = []
            trackers = []
            stored_active_trackers=[t.face_id for t in persons.tracks if t.state=='Confirmed']
            flag, faces = detect_face(frame)  # 提取128维向量，是dlib.vector类的对象
            print("global detect faces",frame_count,len(faces))

            for index, face_position in enumerate(faces):

                person_face, landmarks = get_features(frame, frame_count,face_position)
                match_result=persons._match([person_face],frame_count,mvs_data)
                if len(match_result)==0:
                    continue
                matched_id=match_result[0]

                draw_landmarks(frame, matched_id,face_position, landmarks)
                tracked_face_ids.append(matched_id)
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(face_position[0], face_position[1],face_position[2],face_position[3])
                tracker.start_track(frame, rect)
                trackers.append(tracker)
            for t_id in stored_active_trackers:

                if persons.tracks[t_id].end!=frame_count:
                    print("update the state of confirmed faces-undetected now")
                    roi=persons.tracks[t_id].features[-1].face_area
                    person_face, landmarks = get_features(frame,frame_count,roi)
                    success = persons._update(t_id, person_face, frame_count)
                    if success:
                        draw_landmarks(frame, t_id, roi, landmarks)
            print("end detection")

        else:
           # print('here')
            refresh_timer-=1
            del_ids=[]
            for track_id in range(person_num):
                tracker=trackers[track_id]
                face_id=tracked_face_ids[track_id]
                #check false detected faces
                if persons.tracks[face_id].state == 'Tentative':
                    print("roi detection for the tentative person")
                    pos = tracker.get_position()
                    startX, startY, endX, endY = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
                    flag, faces = detect_face(frame, [startX, startY, endX, endY])
                    if not flag:
                        del_ids.append(track_id)
                        continue
                tracker.update(frame)
                pos = tracker.get_position()
                startX, startY, endX, endY = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
                person_face, landmarks = get_features(frame,frame_count,[startX, startY, endX, endY])
                success = persons._update(face_id, person_face, frame_count)

                if not success:
                    print("there is a track person unmatched")
                    roi_success, faces = detect_face(frame, [startX, startY, endX, endY])
                    # print("ddd",frame_count)
                    if roi_success:
                        print("roi detection", frame_count)
                        face_position = faces[0]
                        person_face, landmarks = get_features(frame,frame_count,face_position)

                        success = persons._update(face_id, person_face, frame_count)
                        tracker = dlib.correlation_tracker()
                        # Start a track on face detected on first frame.
                        rect = dlib.rectangle(face_position[0], face_position[1], face_position[2],
                                              face_position[3])
                        tracker.start_track(frame, rect)
                        trackers[track_id] = tracker
                        draw_landmarks(frame, face_id, face_position, landmarks)
                    else:
                        print("search global (scene change)")
                        frame_success, faces = detect_face(frame)
                        if frame_success:
                            face_position = faces[0]
                            person_face, landmarks = get_features(frame, frame_count,face_position)

                            success = persons._update(face_id, person_face, frame_count)
                            tracker = dlib.correlation_tracker()
                            # Start a track on face detected on first frame.
                            rect = dlib.rectangle(face_position[0], face_position[1], face_position[2],
                                                  face_position[3])
                            tracker.start_track(frame, rect)
                            trackers[track_id] = tracker
                            draw_landmarks(frame, face_id, face_position, landmarks)


                else:
                    draw_landmarks(frame, face_id, [startX, startY, endX, endY], landmarks)

            del_ids.sort(reverse=True)
            for id in del_ids:
                print(len(trackers), len(del_ids), id)
                trackers.pop(id)
                tracked_face_ids.pop(id)

        persons._check_state(frame_count)
        cv2.imshow('Video', frame)
        out.write(frame)
        person_num=len(trackers)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    out.release()
    print("Frame count",frame_count)
    # Closes all the frames
    cv2.destroyAllWindows()
    persons._print()
    return persons

def merge_video(original_video,forged_video,merged_video,insert_len=100):
    cap_ori = cv2.VideoCapture(original_video)
    frame_num_ori = int(cap_ori.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_forged = cv2.VideoCapture(forged_video)
    frame_num_forged = int(cap_forged.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_num_ori, frame_num_forged)

    cut_start = np.random.randint(0, min(frame_num_ori, frame_num_forged) - insert_len, size=1)[0]
    video_fps = cap_ori.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap_ori.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap_ori.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(merged_video, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, video_size, True)
    frame_count=0
    step=25
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
def gen_features_for_video(video_path,output_path):
    person_data = genFeatures(video_path)
    f = open(output_path, "wb")
    person = pickle.dumps(person_data)
    f.write(person)
    f.close()

def get_distance_data(original_lip_info_path, forged_lip_info_path,pickmode='pickall',threshold=10,attr_arr=[2]):

    ori_video_len,forged_video_len=np.load('data/forensics_video_len.npy',allow_pickle=True)
    _, lip_sizes = np.load('data/forensics_lip_size.npy', allow_pickle=True)

    persons_ori = readData(original_lip_info_path)
   # print("persons_ori", np.load(original_lip_info_path, allow_pickle=True))
    persons_deepfake = readData(forged_lip_info_path)
   # print("persons_deepfake", persons_deepfake)

    authen_data_deepfake = get_authen_data(persons_deepfake,persons_ori, lip_sizes[0], ori_video_len[0] / ori_video_len[0])[0][0]

    return authen_data_deepfake
# original_video='D:/TangLi/datasets/FaceForensics++/original_sequences/youtube/c23/videos/000.mp4'
# forged_video='result_voice.mp4'
# merged_video='lip_syncing_merged.mp4'
# tamper_range,cut_start=merge_video(original_video,forged_video,merged_video)
# gen_features_for_video(original_video,'ori_persons_info')
# gen_features_for_video(merged_video,'forged_persons_info')
#
# distance_data=get_distance_data('ori_persons_info','forged_persons_info')
# np.save('distance_merged_lip_synicing.npy',[distance_data,tamper_range,cut_start])

distance_data,tamper_range,cut_start=np.load('distance_merged_lip_synicing.npy',allow_pickle=True)
print(tamper_range,cut_start)
#print([[i,distance_data[i][2]] for i in range(len(distance_data))])
#forged_segment_mean,forged_segment_model=authen_comparision(distance_data)
#print(tamper_range,forged_segment_mean,forged_segment_model)
segment_data=[[0, [0.07457341691583104, 0.07088658170715866]], [1, [0.10426975242431259, 0.08554951397273061]], [2, [0.11414069441314059, 0.0810536136293698]], [3, [0.10768148023134595, 0.06467012254911164]], [4, [0.14369435215790308, 0.09004306699261444]], [5, [0.22152667446803936, 0.13553266314006399]], [6, [0.23434846564733713, 0.14587817947846526]], [7, [0.1513447164872091, 0.12012789499584059]], [8, [0.09914947835122163, 0.06226681926241372]], [9, [0.09808743576817108, 0.060668092172429154]], [10, [0.09162069779729083, 0.05052499631517317]], [11, [0.0924890018921532, 0.053859516821052764]], [12, [0.0847156204092485, 0.060878012908328394]], [13, [0.07419977553401691, 0.055164747630938475]], [14, [0.06020983276793347, 0.031442422170410364]]]
#肉眼观察的话 篡改的其实是125到241
step=25
plt.figure(figsize=(9,4))
plt.plot([distance_data[i][2] for i in range(len(distance_data))],label='distance')
#plt.plot([d[0]*step+step for d in segment_data],[d[1][0] for d in segment_data],linestyle='dotted')
plt.plot([d[0]*step+step for d in segment_data],[d[1][0] for d in segment_data],'ro',linestyle='dotted',markerfacecolor='white',label='mean value')
#plt.plot([index*step+step for index in forged_segment_model],[segment_data[index][1][0] for index in forged_segment_model],'rx')
plt.plot([index*step+step for index in [4,5,6,7]],[segment_data[index][1][0] for index in [4,5,6,7]],'rx')
plt.ylabel('Distance',fontsize=16)
plt.xlabel('Index',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.savefig('partial_lip_syncing_authen.pdf',bbox_inches='tight')
plt.show()