######数据预处理
#读取数据
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
import csv
import numpy as np 
import random

class Utils(object):
    def init(self):
        pass
    def pro_data(self):
        
        w2id , id2w = pickle.load(open('data/w2id_id2w.pkl' , 'rb'))  #词汇表
        poem = pickle.load(open('data/index.pkl' , 'rb'))
        classified_csvpoem = open("data/Classified_Poem.csv", "r", encoding="utf-8-sig")  #分类过的诗的CSV文件
        reader = csv.reader(classified_csvpoem)

        poem_list = []   #存放诗词的数组
        label_list = []  #存放春夏秋冬的标签的数组
        for line in reader:  
            poem_list.append(line[:-1])
            label_list.append(line[33])

        print ('唐诗数据集大小 : {}'.format(len(poem_list)))
        print(len(w2id))
        
        #打乱数据

        seed = 50
        random.seed(seed)
        random.shuffle(poem_list)
        random.seed(seed) #两个list同步打乱
        random.shuffle(label_list)
        
        return poem_list , label_list


        # print(poem_list[:10])
        # print(np.array(poem_list).shape)
        # print(label_list[:10])
        # print(np.array(label_list).shape)

