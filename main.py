import os
import random
import numpy as np
import torch
import argparse

from load_data import load_dataset, split_dataset, allocate_dataset, Dataset_Config
from attack_model import EQB2A

# Locking random seed
def seed_setting(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# seed_setting()

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--dataset', dest='dataset', default='WIKI', choices=['WIKI', 'IAPR', 'FLICKR', 'COCO', 'NUS'])
parser.add_argument('--dataset_path', dest='dataset_path', default='../Datasets/')
# attacked model
parser.add_argument('--method', dest='method', default='DCMH', choices=['DCMH', 'CPAH', 'DADH', 'DJSRH', 'JDSH', 'DGCPN'])
parser.add_argument('--attacked_models_path', dest='attacked_models_path', default='attacked_models/')
# output
parser.add_argument('--output_dir', dest='output_dir', default='output000')
parser.add_argument('--output_path', dest='output_path', default='outputs/')
# detail setting
parser.add_argument('--bit', dest='bit', type=int, default=32)
parser.add_argument('--gpu', dest='gpu', type=str, default='0', choices=['0', '1', '2', '3'])
# knockoff model
parser.add_argument('--knockoff_bit', dest='kb', type=int, default=32)
parser.add_argument('--knockoff_epochs', dest='ke', type=int, default=1)
parser.add_argument('--knockoff_batch_size', dest='kbz', type=int, default=24)
parser.add_argument('--knockoff_text_learning_rate', dest='ktlr', type=float, default=1e-3)
parser.add_argument('--knockoff_image_learning_rate', dest='kilr', type=float, default=1e-4)
# perturbation model
parser.add_argument('--perturbation_epochs', dest='pe', type=int, default=1)
parser.add_argument('--perturbation_batch_size', dest='pbz', type=int, default=24)
parser.add_argument('--perturbation_learning_rate', dest='plr', type=float, default=1e-4)
# attack model
parser.add_argument('--attack_epochs', dest='ae', type=int, default=1)
parser.add_argument('--attack_batch_size', dest='abz', type=int, default=24)
parser.add_argument('--attack_learning_rate', dest='alr', type=float, default=1e-4)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# data processing
Dcfg = Dataset_Config(args.dataset, args.dataset_path)
X, Y, L = load_dataset(Dcfg.data_path)
X, Y, L = split_dataset(X, Y, L, Dcfg.query_size, Dcfg.training_size, Dcfg.database_size)
Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L = allocate_dataset(X, Y, L)
print('Tr_I:', Tr_I.shape, 'Tr_T:', Tr_T.shape, 'Tr_L:', Tr_L.shape)
print('Db_I:', Db_I.shape, 'Db_T:', Db_T.shape, 'Db_L:', Db_L.shape)
print('Te_I:', Te_I.shape, 'Te_T:', Te_T.shape, 'Te_L:', Te_L.shape)

# black-box attack
EQB2A = EQB2A(args, Dcfg)
EQB2A.test_attacked_model(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L)
EQB2A.train_knockoff(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L) # EQB2A.train_knockoff(Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L)
EQB2A.test_knockoff(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L)
EQB2A.train_perturbation_model(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L) # EQB2A.train_perturbation_model(Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L)
EQB2A.test_perturbation_model(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L)
EQB2A.train_attack_model(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L) # EQB2A.train_attack_model(Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L)
EQB2A.test_attack_model(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L)