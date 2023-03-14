import face_rec_video as fc
import os
import cv2
import numpy as np
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

already_exists=False

def moving_object_num(mvs_data,shape):
    mvs_count=0
    threshold=100
    pos_x,pos_y=mvs_data

    bins_x=np.arange(0,shape[0]+128,128)
    hists_x, bins_x =np.histogram(pos_x,bins_x)
    peaks_x, _ = find_peaks(hists_x,distance=5,height=10)#低于height的不考虑


    return len(peaks_x)#max(len(peaks_x),len(peaks_y))
def getFaceArea(image):

    dets = detector(image, 1)
    flag = True
    if len(dets) == 0:
        flag = False
        return flag, []
    # print("dets",self.img_rgb)
    # print("Number of faces detected: {}".format(len(dets)))
    return flag,dets
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
def genFeatures(video_path):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    detect_flag=True
    refresh_timer=100000

    trackers=[]
    frame_count=0
    video_data=[]
    filename, t = os.path.splitext(video_path)
    frames=[]
    while True:
        success, frame = cap.read()
        # print(real_frame)
        if not success:
            break
        if frame_count!=0:
            detect_flag = False

        while True:
            persons = []
            if detect_flag:
                track_id = 0
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
                    track_id += 1
                    frame = cv2.rectangle(frame, (persons[-1][1][0], persons[-1][1][1]),
                                          (persons[-1][1][2], persons[-1][1][3]), (255, 0, 0), 2)

            else:

                for track_id, tracker in enumerate(trackers):
                    tracker.update(frame)
                    pos = tracker.get_position()
                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    face = dlib.rectangle(startX, startY, endX, endY)
                    #    print("pos", [(startX,startY),(endX,endY)], pos.left(), pos.top(), pos.right(), pos.bottom())
                    shape = shape_predictor(frame, face)
                    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
                    # print("face",face)
                    face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
                    # check if activate detect

                    positions_lip_arr = []
                    for idx, point in enumerate(landmarks[48:68]):
                        # 68点的坐标
                        pos = (point[0, 0], point[0, 1])
                        positions_lip_arr.append(pos)

                    persons.append([face_descriptor, [startX, startY, endX, endY], positions_lip_arr])


            video_data.append([frame_count,detect_flag, persons])
            frame_count += 1
            break


    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return video_data

def genForFolderObjectPath(folder_path,key_word='_features_all_track_by_detect_onlytrack.npy'):
    pathes=[]
    multi_person_v=[]
    for file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, file)
        video_path = video_path.replace('\\', '/')


        if '.mp4' in video_path:

            p1 = os.path.basename(video_path)
            file_name = os.path.splitext(p1)[0]
            object_path=folder_path+'/'+file_name+key_word#'_features_all_track.npy'
            print("gen for",object_path)
            pathes.append(object_path)
            if False:#os.path.exists(object_path):
                person_data = np.load(object_path, allow_pickle=True)
            else:
                person_data = genFeatures(video_path)
                np.save(object_path, person_data)

    return pathes


def divide_fake_paths(ori_paths,fake_paths):
    fake_sets=[]
    for i in range(len(ori_paths)):
        sub_fake_paths=[]
        p1 = os.path.basename(ori_paths[i])
        file_name = (os.path.splitext(p1)[0]).replace('_features_all_track_by_detect_onlytrack', '')
        d=str.split(file_name,'_')
        print('ori',file_name)
        for path in fake_paths:
            if d[0] in path and d[1] in path:
                sub_fake_paths.append(path)
                print("fake",path)
        fake_sets.append(fake_sets)
    return fake_sets

def traverse_faceforensics_videos():
    data_path_real = 'D:/TangLi/datasets/FaceForensics++/original_sequences/youtube/c23/videos'
    data_path_com1 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf20'
    data_path_com2 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf30'
    data_path_com3 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf40'
    data_path_forged00='D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/c23/videos'
    data_path_forged0 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf20'
    data_path_forged1 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf30'
    data_path_forged2 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf40'
    data_path_com = [data_path_com1, data_path_com2, data_path_com3]
    data_path_forged=[data_path_forged0,data_path_forged1,data_path_forged2,data_path_forged00]
    key_word = '_features_all_detection_free_track.npy'

    ori_paths=genForFolderObjectPath(data_path_real,key_word)

    return

traverse_faceforensics_videos()
#mul_person_video=np.load('celeb_v2_multi_person_video.npy',allow_pickle=True)
