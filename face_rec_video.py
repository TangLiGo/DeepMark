import sys
import dlib
import cv2
import os
import glob
import numpy as np

def comparePersonData(data1, data2):
    diff = 0
    # for v1, v2 in data1, data2:
        # diff += (v1 - v2)**2
    for i in range(len(data1)):
        diff += (data1[i] - data2[i])**2
    diff = np.sqrt(diff)
   # print( diff)
    if(diff < 0.6):
    #    print( "It's the same person")
        return 1
    else:
     #   print ("It's not the same person")
        return 0
def getDistance(data1, data2):
    diff = 0
    # for v1, v2 in data1, data2:
        # diff += (v1 - v2)**2
    for i in range(len(data1)):
        diff += (data1[i] - data2[i])**2
    diff = np.sqrt(diff)
   # print( diff)
    return diff
def savePersonData(face_rec_class, face_descriptor):
    if face_rec_class.name == None or face_descriptor == None:
        return
    filePath = face_rec_class.dataPath + face_rec_class.name + '.npy'
    vectors = np.array([])
    for i, num in enumerate(face_descriptor):
        vectors = np.append(vectors, num)
        # print(num)
   # print('Saving files to :'+filePath)
 #   np.save(filePath, vectors)
    return vectors

def loadPersonData(face_rec_class, personName):
    if personName == None:
        return
    filePath = face_rec_class.dataPath + personName + '.npy'
    vectors = np.load(filePath)
   # print(vectors)
    return vectors

class face_recognition(object):
    def __init__(self):
        self.current_path = os.getcwd() # 获取当前路径
        #加载特征点识别模型
        self.predictor_path = "C:/Users/Tangli/Downloads/codes/shape_predictor_68_face_landmarks.dat"
        self.shape_predictor = dlib.shape_predictor(self.predictor_path)
        #加载人脸识别模型
        self.face_rec_model_path =  "C:/Users/Tangli/Downloads/codes/dlib_face_recognition_resnet_model_v1.dat"
        self.face_rec_model = dlib.face_recognition_model_v1(self.face_rec_model_path)

        self.faces_folder_path = self.current_path + "\\faces\\"
        self.dataPath = self.current_path + "\\data\\"
        #正向人脸检测器
        self.detector = dlib.get_frontal_face_detector()


        self.name = None
        self.img_bgr = None
        self.img_rgb = None



    def inputPerson(self, name='people', img=None):
        #if img == None:
         #   print('No file!\n')
         #   return

        # img_name += self.faces_folder_path + img_name
        self.name = name
       # self.img_bgr = cv2.imread(self.current_path+img_path)
        self.img_bgr = img
        # opencv的bgr格式图片转换成rgb格式
        b, g, r = cv2.split(self.img_bgr)
        self.img_rgb = cv2.merge([r, g, b])

    def create128DVectorSpace(self):
        dets = self.detector(self.img_rgb, 1)
        flag=True
        if len(dets)==0:
            flag=False
            return flag, []
       # print("dets",self.img_rgb)
       # print("Number of faces detected: {}".format(len(dets)))
        for index, face in enumerate(dets):
            #print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))

            shape = self.shape_predictor(self.img_rgb, face)
           # print("face",face)
            face_descriptor = self.face_rec_model.compute_face_descriptor(self.img_rgb, shape)

            # print(face_descriptor)
            # for i, num in enumerate(face_descriptor):
            #   print(num)
            #   print(type(num))

            return flag, face_descriptor

