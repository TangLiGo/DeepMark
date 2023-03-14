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

def moving_object_num(i,mvs_path,shape):
    mvs_count=0
    threshold=100

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

def genFeatures(video_path):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output = cv2.VideoWriter('169.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), video_fps,
                             video_size)
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

        b, g, r = cv2.split(frame)
        img_rgb = cv2.merge([r, g, b])
        #detect_flag = True

        if frame_count!=0:

            if refresh_timer==0:
                detect_flag = True
             #   print("frame{} needs face detection".format(frame_count))
                refresh_timer = video_fps
            else:
                detect_flag = False
                refresh_timer -= 1
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


            cv2.imwrite('refresher_imgs/' + str(frame_count) + '.png',frame)
            output.write(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            # Else we just attempt to track from the previous frame
            # track all the detected faces
            for tracker in trackers:
                tracker.update(frame)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                face=dlib.rectangle(startX,startY,endX,endY)
            #    print("pos", [(startX,startY),(endX,endY)], pos.left(), pos.top(), pos.right(), pos.bottom())
                shape = shape_predictor(frame, face)
                landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
                # print("face",face)
                face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
                positions_lip_arr = []
                for idx, point in enumerate(landmarks[48:68]):
                    # 68点的坐标
                    pos = (point[0, 0], point[0, 1])
                    positions_lip_arr.append(pos)

                persons.append([face_descriptor, [startX,startY,endX,endY], positions_lip_arr])

            output.write(frame)
            cv2.imwrite('refresher_imgs/' + str(frame_count) + '.png', frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        video_data.append([frame_count, persons])
        frame_count += 1

    cap.release()
    output.release()
    # Closes all the frames
    cv2.destroyAllWindows()

    return video_data
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
def genForFolderObjectPath(folder_path,key_word='_features_all_track.npy'):
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
            if os.path.exists(object_path):
                person_data = np.load(object_path, allow_pickle=True)
            else:
                person_data = genFeatures(video_path)
                #np.save(object_path, person_data)
            d,y=check_multi_people(person_data)
            if y:
                multi_person_v.append(object_path)
    return pathes,multi_person_v


def divide_fake_paths(ori_paths,fake_paths):
    fake_sets=[]
    for i in range(len(ori_paths)):
        sub_fake_paths=[]
        p1 = os.path.basename(ori_paths[i])
        file_name = (os.path.splitext(p1)[0]).replace('_features_all_track', '')
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
    key_word = '_features_all_track.npy'
    multi_person_v=[]
    ori_paths,multi_person_v1=genForFolderObjectPath(data_path_real,key_word)
    multi_person_v=multi_person_v+multi_person_v1
    # for data_folder in data_path_com:
    #     crf_paths, multi_person_v1 = genForFolderObjectPath(data_folder, key_word)
    #     multi_person_v=multi_person_v+multi_person_v1
    # for data_folder in data_path_forged:
    #     fake_paths, multi_person_v1 = genForFolderObjectPath(data_folder, key_word)
    #     multi_person_v=multi_person_v+multi_person_v1
  #  fake_paths_raw,multi_person_v2=genForFolderObjectPath(data_path_forged1,key_word)
   # fake_paths=divide_fake_paths(ori_paths,fake_paths_raw)
   # #np.save('data/faceforensics_multi_person_video_crf_notrack.npy', multi_person_v)
    return multi_person_v
    # for folder in data_path_forged:
    #     for subfolder in os.listdir(folder):
    #         sub_file_path = os.path.join(folder, subfolder)
    #         sub_file_path = sub_file_path.replace('\\', '/')
    #         genForFolderObjectPath(sub_file_path)

    # for path in data_path_com:
    #     genForFolderObjectPath(path)

#print(np.load('celeb_v2_multi_person_video.npy',allow_pickle=True))
# data1=np.load('data/faceforensics_multi_person_video_crf_notrack.npy', allow_pickle=True)
# data2=np.load("data/faceforensics_multi_person_video_face2face.npy", allow_pickle=True)
# print("data1",data1)
# print("data2",data2)
# print("data2.extend(data1)",data2.extend(data1))
##np.save('data/faceforensics_multi_person_video_face2face.npy', data2.extend(data1))
#traverse_faceforensics_videos()
genFeatures('D:\TangLi\datasets\Celeb-DF\Celeb-real/id7_0000.mp4')
#genFeatures('D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf20/169.mp4')
#genFeatures('D:/TangLi/datasets/FaceForensics++/original_sequences/youtube\c23/videos/908.mp4')#212
#genFeatures('D:\TangLi\datasets\Celeb-DF\Celeb-real/id10_0007.mp4')
#mul_person_video=np.load('celeb_v2_multi_person_video.npy',allow_pickle=True)
fa=[31,57,88,164,170,181,182,204,212,213,255,266,328,333,396,541,545,614,728,759,764,776,781,889,909,953]