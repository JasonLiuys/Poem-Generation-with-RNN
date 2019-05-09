#######组织数据集
import numpy as np 
import utils
import pickle

BATCH_SIZE = 64

utils = utils.Utils()
poem_list, label_list = utils.pro_data()
w2id , id2w = pickle.load(open('data/w2id_id2w.pkl' , 'rb'))  #词汇表

class Dataset(object):
    def __init__(self , batch_size):
        self.batch_size = batch_size
        self.data, self.target = self.read_data()
        self.start = 0
        self.lenth = len(self.data)  #一共有多少首诗
        
    def read_data(self):
        id_list = []
        for poem in poem_list:
            id_list.append([w2id[idx] for idx in poem])
        batch_num = len(id_list) // self.batch_size
        # data和target
        x_data = []
        y_data = []
        # 生成batch
        for i in range(batch_num):
            # 截取一个batch的数据
            start = i * self.batch_size
            end = start + self.batch_size
            batch = id_list[start:end]
            # 计算最大长度
            max_lenth = max(map(len, batch))
            # 填充
            tmp_x = np.full((self.batch_size, max_lenth), 0, dtype=np.int32)
            # 数据覆盖
            for row in range(self.batch_size):
                tmp_x[row, :len(batch[row])] = batch[row]  #矩阵{3635,64,33}
            tmp_y = np.copy(tmp_x)
            tmp_y[:, :-1] = tmp_y[:, 1:]  #data与target错位一个
            x_data.append(tmp_x)
            y_data.append(tmp_y)
#         print(np.array(x_data).shape)
        return x_data, y_data   #所有数据
    
    def next_batch(self):
        start = self.start
        self.start += 1  #下一个batch
        if self.start >= self.lenth:
            self.start = 0
        print(np.array(self.data[start]).shape)
        return self.data[start], self.target[start]

if __name__ == '__main__':
    dataset = Dataset(BATCH_SIZE)
    dataset.read_data()