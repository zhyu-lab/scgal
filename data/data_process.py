import numpy as np
import torch
import math
import copy

class data_set:
    def __init__(self,opt):
        self.opt = opt
        self.A_data,self.A_dim= load_dataDNA(opt.dir_A)  # load data from 'path'
        self.B_data = load_dataRNA(opt.dir_B,self.A_dim) # load data from 'path'
    def __getitem__(self, item):
        A = self.A_data
        B = self.B_data
        A_path = self.dir_A
        B_path = self.dir_B
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

def xs_gen(cell_list, batch_size, random1):
    data_copy = copy.deepcopy(cell_list.dataset)
    data1 = {'A': data_copy.A_data, 'B': data_copy.B_data}
    data_A_len = len(data_copy.A_data)
    data_B_len = len(data_copy.B_data)
    steps = math.ceil(data_A_len / batch_size)
    A_batch_size = batch_size
    B_batch_size = math.ceil(data_B_len/A_batch_size/steps)
    if (A_batch_size*B_batch_size*steps-data_B_len) > (A_batch_size*B_batch_size):
        B_batch_size = B_batch_size-1

    if random1 == 1:
        data_copy.A_data = data_copy.A_data.numpy()
        data_copy.B_data = data_copy.B_data.numpy()
        list_norm = np.sum(data_copy.B_data,axis=1)
        norm_med = np.median(list_norm)
        for i in range(data_copy.B_data.shape[0]):  # normalize for library size
            data_copy.B_data[i, :] = data_copy.B_data[i, :] / (np.sum(data_copy.B_data[i, :])/norm_med)
        data_copy.B_data = RNA_processing_log(data_copy.B_data)
        np.random.shuffle(data_copy.A_data)
        np.random.shuffle(data_copy.B_data)
        data_copy.A_data = torch.from_numpy(data_copy.A_data)
        data_copy.B_data = torch.from_numpy(data_copy.B_data)
    elif random1 == 0:
        data_copy.A_data = data_copy.A_data.numpy()
        data_copy.B_data = data_copy.B_data.numpy()
        list_norm_test = np.sum(data_copy.B_data, axis=1)
        norm_med_test = np.median(list_norm_test)
        for i in range(data_copy.B_data.shape[0]):  # normalize for library size
            data_copy.B_data[i, :] = data_copy.B_data[i, :] / (np.sum(data_copy.B_data[i, :])/norm_med_test)
        data_copy.B_data = RNA_processing_log(data_copy.B_data)
        data_copy.A_data = torch.from_numpy(data_copy.A_data)
        data_copy.B_data = torch.from_numpy(data_copy.B_data)
    for i in range(steps):
        batch_x = data_copy.A_data[i * A_batch_size: i * A_batch_size + A_batch_size]
        k = A_batch_size
        input_dataB = k*B_batch_size
        batch_y = data_copy.B_data[i * input_dataB: i * input_dataB + input_dataB]
        data1['A'] = batch_x
        data1['B'] = batch_y
        yield i, data1

def load_dataDNA(dir):
    data_o = np.loadtxt(dir, dtype='float32', delimiter=',')
    data_o = DNA_processing_log(data_o)
    s = 0
    b = [True] * data_o.shape[1]
    print("Data loaded from", dir)
    print('dimensions of the data: ', data_o.shape)
    for i in range(data_o.shape[1] - 1):
        if np.array_equal(data_o[:, i], data_o[:, i + 1]) == True:
            s = s + 1
            b[i + 1] = False
    data_1 = data_o[:, b]
    print("scDNA-seq data for the first step of feature selection：", data_1.shape)
    tv = np.var(data_1, axis=0) != 0
    data = data_1[:, tv]
    print('scDNA-seq after filtering, the dimensions reduce to: ', data.shape)
    data = cv_Dim_reduction(data, 1024)
    print('Dimensionality reduction based on coefficient of variation, the dimensions reduce to: ', data.shape)
    a_dim=data.shape[1]
    data= torch.from_numpy(data)
    return data,a_dim

def load_dataRNA(dir,a_dim):
    data_o = np.loadtxt(dir, dtype='float32', delimiter=',')
    s = 0
    b = [True] * data_o.shape[1]
    print("Data loaded from", dir)
    print('dimensions of the data: ', data_o.shape)
    for i in range(data_o.shape[1] - 1):
        if np.array_equal(data_o[:, i], data_o[:, i + 1]) == True:
            s = s + 1
            b[i + 1] = False
    data_1 = data_o[:, b]
    print("scRNA-seq data for the first step of feature selection：", data_1.shape)
    tv = np.var(data_1, axis=0) != 0
    data = data_1[:, tv]
    print('scRNA-seq after filtering, the dimensions reduce to: ', data.shape)
    data = cv_Dim_reduction(data,a_dim)
    print('Dimensionality reduction based on coefficient of variation, the dimensions reduce to: ', data.shape)
    data= torch.from_numpy(data)
    return data

def DNA_processing_log(DNA_data):
    index = (DNA_data == 0.0)
    DNA_data[index] = 1
    data = np.log2(DNA_data)
    return data

def RNA_processing_log(p_B_data):
    p_B_data=p_B_data+1
    p_B_data=np.log2(p_B_data)
    return p_B_data

def cv_Dim_reduction(data,a_dim): # Dimensionality reduction based on coefficient of variation
    data_cv = copy.deepcopy(data)
    b = [False] * data_cv.shape[1]
    avg = np.mean(data_cv, axis=0) # Calculate the average value
    var = np.var(data_cv, axis=0) # Calculating the variance
    cv = var / avg  # Calculate the coefficient of variation
    cv = cv[np.argsort(-cv)]
    cv = cv[0:a_dim]
    for i in range(data_cv.shape[1]):
        temp = np.var(data_cv[:, i]) / np.mean(data_cv[:, i])
        if temp in cv:
            b[i] = True
    data_cv = data_cv[:, b]
    return data_cv