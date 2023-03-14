import numpy as np
import matplotlib.pyplot as plt
import os
def euclidean_distance(a, b):
    # 用于计算成对的平方距离
    # a NxM 代表N个对象，每个对象有M个数值作为embedding进行比较
    # b LxM 代表L个对象，每个对象有M个数值作为embedding进行比较
    # 返回的是NxL的矩阵，比如dist[i][j]代表a[i]和b[j]之间的平方和距离
    # 实现见：https://blog.csdn.net/frankzd/article/details/80251042
    a, b = np.asarray(a), np.asarray(b)  # 拷贝一份数据
    if len(a) == 0 or len(b) == 0:
       # #print("ec", len(np.zeros((len(a), len(b)))))
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(
        b).sum(axis=1)  # 求每个embedding的平方和
    # sum(N) + sum(L) -2 x [NxM]x[MxL] = [NxL]
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    #返回的矩阵 每一行i对应的是a[i] 与各个b[j] j=1-m 的距离

    return np.sqrt(r2)

def check_multi_people(data,th=0.6):
    distances=[]
    flag=False
    for i in range(len(data)-1):
        if len(data[i+1][1])==0 or len(data[i][1])==0:
            continue
      #  print(data[i][1])
        dis=euclidean_distance([data[i][1][0][0]],[data[i+1][1][0][0]])[0,0]
        distances.append(dis)
        if dis>th:
            flag=True
            #break
    plt.figure()
    plt.plot(distances)
    plt.show()
    return flag

# compression_level=0
# data=np.load("data/complevel"+str(compression_level)+"_mul_people_videos.npy",allow_pickle=True)
# print(data)
# id=5
# check_multi_people(np.load(data[id],allow_pickle=True))
# print(len(data)/2)
#
# print(np.load("D:\\TangLi\\datasets\\FaceForensics++\\manipulated_sequences\\Face2Face\\Lossy_crf20\\001_870_lip_all_info_v0.npy",allow_pickle=True)[0])
# print(np.load("D:\\TangLi\\datasets\\FaceForensics++\\original_sequences\\youtube\\c23\\videos\\255_features_all.npy",allow_pickle=True)[0])
def getObjectPaths(data_path,key_word):
    object_path=[]
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        file_path = file_path.replace('\\', '/')

        if key_word in file_path :#and '_track' not in file_path:
            # print("file_path", file_path)
           # lip_info = np.load(video_path, allow_pickle=True)
            os.remove(file_path)
    return object_path



def traverse_forensics_videos(compression_level=0):
    data_path_real = 'D:/TangLi/datasets/FaceForensics++/original_sequences/youtube/c23/videos'
    data_path_com1 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf20'
    data_path_com2 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf30'
    data_path_com3 = 'D:/TangLi/datasets/FaceForensics++/original_sequences_crf40'

    data_path_forged0 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf20'
    data_path_forged1 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf30'
    data_path_forged2 = 'D:/TangLi/datasets/FaceForensics++/manipulated_sequences/Face2Face/Lossy_crf40'
    data_path_com = [data_path_com1, data_path_com2, data_path_com3]
    data_path_forged=[data_path_forged0,data_path_forged1,data_path_forged2]
   # original_lip_info_path=getObjectPaths(data_path_real,'features_all_track')
    #forged_lip_info_path=getObjectPaths(data_path_forged[compression_level],'features_all_track')
    for compression_level in [0,1,2]:
        crf_lip_info_path = getObjectPaths(data_path_com[compression_level], 'v0')

  #  print("forged_lip_info_path",forged_lip_info_path)
    return
traverse_forensics_videos()