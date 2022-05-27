import torch
import os
import errno
from torch.autograd import Variable
import numpy as np

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

def calc_hamming(B1, B2):
    num = B1.shape[0]
    q = B1.shape[1]
    result = torch.zeros(num).cuda()
    for i in range(num):
        result[i] = 0.5 * (q - B1[i].dot(B2[i]))
    return result

def return_samples(index, qB, rB, k=None):
    num_query = qB.shape[0]
    if k is None:
        k = rB.shape[0]
    index_matrix = torch.zeros(num_query, k + 1).int()
    index = torch.from_numpy(index)
    for i in range(num_query):
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        index_matrix[i] = torch.cat((index[i].unsqueeze(0), ind[np.linspace(0, rB.shape[0]-1, k).astype('int')]), 0)
    return index_matrix

def return_results(index, qB, rB, s=None, o=None):
    num_query = qB.shape[0]
    index_matrix = torch.zeros(num_query, 1+s+o).int()
    index = torch.from_numpy(index)
    for i in range(num_query):
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        index_matrix[i] = torch.cat((index[i].unsqueeze(0), ind[:s], ind[np.linspace(0, rB.shape[0]-1, o).astype('int')]), 0)
    return index_matrix

def CalcMap(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    return S

def log_trick(x):
    lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt

def image_normalization(_input):
    _input = 2 * _input / 255 - 1
    return _input

def image_restoration(_input):
    _input = (_input + 1) / 2 * 255
    return _input

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise