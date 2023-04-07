import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import os
from scipy.special import comb

from attacked_model import Attacked_Model
from models import SurrogateImageModel, SurrogateTextModel, PerturbationGenerator, PerturbationSupervisor, ImageGenerator, Discriminator, GANLoss
from utils import mkdir_p, calc_hamming, CalcMap, image_normalization, image_restoration, return_results

class EQB2A(nn.Module):
    def __init__(self, args, Dcfg):
        super(EQB2A, self).__init__()
        self.args = args
        self.Dcfg = Dcfg
        self._build_model()
        self._save_setting()
    
    def _build_model(self):
        self.attacked_model = Attacked_Model(self.args.method, self.args.dataset, self.args.bit, self.args.attacked_models_path, self.args.dataset_path)
        self.attacked_model.eval().cuda()
        self.SIM = SurrogateImageModel(self.args.bit).train().cuda()
        self.STM = SurrogateTextModel(self.Dcfg.tag_dim, self.args.bit).train().cuda()
        self.PG = PerturbationGenerator().train().cuda()
        self.PS = PerturbationSupervisor(self.args.bit).train().cuda()
        self.IG = ImageGenerator().train().cuda()
        self.D = Discriminator().train().cuda()
        self.criterionGAN = GANLoss().cuda()

    def _save_setting(self):
        self.output = os.path.join(self.args.output_path, self.args.output_dir)
        self.model_dir = os.path.join(self.output, 'Model')
        self.image_dir = os.path.join(self.output, 'Image')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)

    def save_surrogate_model(self):
        torch.save(self.SIM.state_dict(), os.path.join(self.model_dir, 'surrogate_imagenet.pth'))
        torch.save(self.STM.state_dict(), os.path.join(self.model_dir, 'surrogate_textnet.pth'))

    def save_perturbation_model(self):
        torch.save(self.PG.state_dict(), os.path.join(self.model_dir, 'perturbation_generator.pth'))
        torch.save(self.PS.state_dict(), os.path.join(self.model_dir, 'perturbation_supervisor.pth'))

    def save_attack_model(self):
        torch.save(self.IG.state_dict(), os.path.join(self.model_dir, 'image_generator.pth'))
        torch.save(self.D.state_dict(), os.path.join(self.model_dir, 'discriminator.pth'))

    def load_surrogate_model(self):
        self.SIM.load_state_dict(torch.load(os.path.join(self.model_dir, 'surrogate_imagenet.pth')))
        self.STM.load_state_dict(torch.load(os.path.join(self.model_dir, 'surrogate_textnet.pth')))
        self.SIM.eval().cuda()
        self.STM.eval().cuda()

    def load_perturbation_model(self):
        self.PG.load_state_dict(torch.load(os.path.join(self.model_dir, 'perturbation_generator.pth')))
        self.PS.load_state_dict(torch.load(os.path.join(self.model_dir, 'perturbation_supervisor.pth')))
        self.PG.eval().cuda()
        self.PS.eval().cuda()

    def load_attack_model(self):
        self.IG.load_state_dict(torch.load(os.path.join(self.model_dir, 'image_generator.pth')))
        self.IG.eval().cuda()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def test_attacked_model(self, Te_I, Te_T, Te_L, Db_I, Db_T, Db_L):
        print('test attacked model...')
        IqB = self.attacked_model.generate_image_hashcode(Te_I)
        TqB = self.attacked_model.generate_text_hashcode(Te_T)
        IdB = self.attacked_model.generate_image_hashcode(Db_I)
        TdB = self.attacked_model.generate_text_hashcode(Db_T)
        I2T_map = CalcMap(IqB, TdB, Te_L, Db_L, 50)
        T2I_map = CalcMap(TqB, IdB, Te_L, Db_L, 50)
        I2I_map = CalcMap(IqB, IdB, Te_L, Db_L, 50)
        T2T_map = CalcMap(TqB, TdB, Te_L, Db_L, 50)
        print('I2T@50: {:.4f}'.format(I2T_map))
        print('T2I@50: {:.4f}'.format(T2I_map))
        print('I2I@50: {:.4f}'.format(I2I_map))
        print('T2T@50: {:.4f}'.format(T2T_map))

    def train_knockoff(self, Tr_I, Tr_T, Tr_L):
        print('train knockoff...')
        query_sampling_number = 2000
        near_sample_number = 5
        rank_sample_number = 5
        optimizer_STM = torch.optim.Adam(self.STM.parameters(), lr=self.args.ktlr, betas=(0.5, 0.999))
        optimizer_SIM = torch.optim.Adam(filter(lambda p: p.requires_grad, self.SIM.parameters()), lr=self.args.kilr, betas=(0.5, 0.999))
        index_FS_T = np.random.choice(range(Tr_T.size(0)), query_sampling_number, replace = False)
        index_FS_I = np.random.choice(range(Tr_I.size(0)), query_sampling_number, replace = False)
        qTB = self.attacked_model.generate_text_hashcode(Tr_T[index_FS_T].type(torch.float).cuda())
        qIB = self.attacked_model.generate_image_hashcode(Tr_I[index_FS_I].type(torch.float).cuda())
        dTB = self.attacked_model.generate_text_hashcode(Tr_T)
        dIB = self.attacked_model.generate_image_hashcode(Tr_I)
        index_matrix_before_TI = return_results(index_FS_T, qTB, dIB, near_sample_number, rank_sample_number)
        index_matrix_before_IT = return_results(index_FS_I, qIB, dTB, near_sample_number, rank_sample_number)
        train_sample_numbers = query_sampling_number * comb(rank_sample_number, 2).astype(int)
        index_matrix_after_TI = np.zeros((train_sample_numbers, 4), int)
        index_matrix_after_IT = np.zeros((train_sample_numbers, 4), int)
        line = 0
        for i in range(query_sampling_number):
            for j in range(near_sample_number+1, near_sample_number+rank_sample_number):
                for k in range(j+1, near_sample_number+rank_sample_number+1):
                    index_matrix_after_TI[line, :3] = index_matrix_before_TI[i, [0, j, k]]
                    index_matrix_after_TI[line, 3] = k-j
                    index_matrix_after_IT[line, :3] = index_matrix_before_IT[i, [0, j, k]]
                    index_matrix_after_IT[line, 3] = k-j
                    line = line + 1
        ranking_loss = torch.nn.MarginRankingLoss(margin=0.1)
        for epoch in range(self.args.ke):
            index = np.random.permutation(train_sample_numbers)
            for i in range(train_sample_numbers // self.args.kbz + 1):
                optimizer_STM.zero_grad()
                optimizer_SIM.zero_grad()
                end_index = min((i+1)*self.args.kbz, train_sample_numbers)
                num_index = end_index - i*self.args.kbz
                ind = index[i*self.args.kbz : end_index]

                anchor_TI = self.STM(Tr_T[index_matrix_after_TI[ind, 0]].type(torch.float).cuda())
                rank1_TI = self.SIM(Tr_I[index_matrix_after_TI[ind, 1]].type(torch.float).cuda())
                rank2_TI = self.SIM(Tr_I[index_matrix_after_TI[ind, 2]].type(torch.float).cuda())
                ranking_target_TI = - 1. / torch.from_numpy(index_matrix_after_TI[ind, 3]).type(torch.float).cuda() # ranking_target_TI = - torch.ones(num_index).cuda()
                hamming_rank1_TI = calc_hamming(anchor_TI, rank1_TI) / self.args.bit
                hamming_rank2_TI = calc_hamming(anchor_TI, rank2_TI) / self.args.bit
                rank_loss_TI = ranking_loss(hamming_rank1_TI.cuda(), hamming_rank2_TI.cuda(), ranking_target_TI)
                quant_loss_TI = (torch.sign(anchor_TI) - anchor_TI).pow(2).sum() / (self.args.bit * num_index)
                balan_loss_TI = (torch.eye(self.args.bit).cuda() * ((torch.ones(num_index, self.args.bit).T.cuda()).mm(anchor_TI))).pow(2).sum() / (self.args.bit * num_index)

                anchor_IT = self.SIM(Tr_I[index_matrix_after_IT[ind, 0]].type(torch.float).cuda())
                rank1_IT = self.STM(Tr_T[index_matrix_after_IT[ind, 1]].type(torch.float).cuda())
                rank2_IT = self.STM(Tr_T[index_matrix_after_IT[ind, 2]].type(torch.float).cuda())
                ranking_target_IT = - 1. / torch.from_numpy(index_matrix_after_IT[ind, 3]).type(torch.float).cuda() # ranking_target_IT = - torch.ones(num_index).cuda()
                hamming_rank1_IT = calc_hamming(anchor_IT, rank1_IT) / self.args.bit
                hamming_rank2_IT = calc_hamming(anchor_IT, rank2_IT) / self.args.bit
                rank_loss_IT = ranking_loss(hamming_rank1_IT.cuda(), hamming_rank2_IT.cuda(), ranking_target_IT)
                quant_loss_IT = (torch.sign(anchor_IT) - anchor_IT).pow(2).sum() / (self.args.bit * num_index)
                balan_loss_IT = (torch.eye(self.args.bit).cuda() * ((torch.ones(num_index, self.args.bit).T.cuda()).mm(anchor_IT))).pow(2).sum() / (self.args.bit * num_index)

                alpha = 0.01
                beta = 0.001
                loss_K = rank_loss_TI + rank_loss_IT + alpha * (balan_loss_TI + balan_loss_IT) + beta * (quant_loss_TI + quant_loss_IT)
                loss_K.backward()
                optimizer_STM.step()
                optimizer_SIM.step()
            print('epoch:{:2d}    loss_K:{:.4f}  rank_loss_TI:{:.4f}  rank_loss_IT:{:.4f}  balan_loss_TI:{:.4f}  balan_loss_IT:{:.4f}  quant_loss_TI:{:.4f}  quant_loss_IT:{:.4f}'
                .format(epoch, loss_K, rank_loss_TI, rank_loss_IT, balan_loss_TI, balan_loss_IT, quant_loss_TI, quant_loss_IT))
        self.save_surrogate_model()
        print('train knockoff done.')

    def test_knockoff(self, Te_I, Te_T, Te_L, Db_I, Db_T, Db_L):
        print('test surrogate model...')
        self.load_surrogate_model()
        IqB = self.SIM.generate_hash_code(Te_I)
        TqB = self.STM.generate_hash_code(Te_T)
        IdB = self.SIM.generate_hash_code(Db_I)
        TdB = self.STM.generate_hash_code(Db_T)
        I2T_map = CalcMap(IqB, TdB, Te_L, Db_L, 50)
        T2I_map = CalcMap(TqB, IdB, Te_L, Db_L, 50)
        I2I_map = CalcMap(IqB, IdB, Te_L, Db_L, 50)
        T2T_map = CalcMap(TqB, TdB, Te_L, Db_L, 50)
        print('I2T@50: {:.4f}'.format(I2T_map))
        print('T2I@50: {:.4f}'.format(T2I_map))
        print('I2I@50: {:.4f}'.format(I2I_map))
        print('T2T@50: {:.4f}'.format(T2T_map))

    def train_perturbation_model(self, Tr_I, Tr_T, Tr_L):
        print('train perturbation model...')
        query_sampling_number = 2000
        near_sample_number = 5
        rank_sample_number = 5
        index_FS_I = np.random.choice(range(Tr_I.size(0)), query_sampling_number, replace = False)
        qIB = self.attacked_model.generate_image_hashcode(Tr_I[index_FS_I].type(torch.float).cuda())
        dTB = self.attacked_model.generate_text_hashcode(Tr_T)
        index_matrix_before_IT = return_results(index_FS_I, qIB, dTB, near_sample_number, rank_sample_number)
        train_sample_numbers = query_sampling_number * comb(rank_sample_number, 2).astype(int)
        index_matrix_after_IT = np.zeros((train_sample_numbers, near_sample_number+3), int)
        line = 0
        for i in range(query_sampling_number):
            for j in range(near_sample_number+1, near_sample_number+rank_sample_number):
                for k in range(j+1, near_sample_number+rank_sample_number+1):
                    index_matrix_after_IT[line, :near_sample_number+1] = index_matrix_before_IT[i, :near_sample_number+1]
                    index_matrix_after_IT[line, near_sample_number+1:near_sample_number+3] = index_matrix_before_IT[i, [j, k]]
                    line = line + 1
        self.load_surrogate_model()
        optimizer_P = torch.optim.Adam([{'params': self.PG.parameters()}, {'params': self.PS.parameters()}], lr=self.args.plr, betas=(0.5, 0.999))
        for epoch in range(self.args.pe):
            index = np.random.permutation(train_sample_numbers)
            for i in range(train_sample_numbers // self.args.pbz + 1):
                end_index = min((i+1)*self.args.pbz, train_sample_numbers)
                num_index = end_index - i*self.args.pbz
                ind = index[i*self.args.pbz : end_index]
                batch_image = image_normalization(Tr_I[index_matrix_after_IT[ind, 0]].type(torch.float).cuda())
                optimizer_P.zero_grad()
                batch_perturbation = self.PG(batch_image)
                batch_supervisor_code = self.PS(batch_perturbation)
                
                # P2T_alienation_loss
                alienation_loss_IT = torch.nn.MarginRankingLoss(margin=0.5)
                P2T_alienation_loss = .0
                for j in range(near_sample_number):
                    batch_text_code = self.STM(Tr_T[index_matrix_after_IT[ind, 1+j]].type(torch.float).cuda())
                    hamming_dist_IT = calc_hamming(batch_supervisor_code, batch_text_code) / self.args.bit
                    P2T_alienation_loss = P2T_alienation_loss + alienation_loss_IT(hamming_dist_IT.cuda(), torch.zeros(num_index).cuda(), torch.ones(num_index).cuda())
                
                # P2T_ranking_loss
                ranking_loss_IT = torch.nn.MarginRankingLoss(margin=0.1)
                batch_text_1_code = self.STM(Tr_T[index_matrix_after_IT[ind, near_sample_number+1]].type(torch.float).cuda())
                batch_text_2_code = self.STM(Tr_T[index_matrix_after_IT[ind, near_sample_number+2]].type(torch.float).cuda())
                hamming_dist1_IT = calc_hamming(batch_supervisor_code, batch_text_1_code) / self.args.bit
                hamming_dist2_IT = calc_hamming(batch_supervisor_code, batch_text_2_code) / self.args.bit
                P2T_ranking_loss = ranking_loss_IT(hamming_dist1_IT.cuda(), hamming_dist2_IT.cuda(), torch.ones(num_index).cuda())
                
                # quantization_loss
                quantization_loss = (torch.sign(batch_supervisor_code) - batch_supervisor_code).pow(2).sum() / (self.args.bit * num_index)

                # total loss
                loss = P2T_ranking_loss + 0.001 * quantization_loss + P2T_alienation_loss
                loss.backward()
                optimizer_P.step()
            print('epoch:{:2d}   loss:{:.4f}  P2T_alienation_loss:{:.4f}  P2T_ranking_loss:{:.4f}  quantization_loss:{:.4f}'
                .format(epoch, loss, P2T_alienation_loss, P2T_ranking_loss, quantization_loss))
        self.save_perturbation_model()
        print('train perturbation model done.')

    def test_perturbation_model(self, Te_I, Te_T, Te_L, Db_I, Db_T, Db_L):
        print('test perturbation model...')
        self.load_surrogate_model()
        self.load_perturbation_model()
        qB = torch.zeros([self.Dcfg.query_size, self.args.bit])
        for i in range(self.Dcfg.query_size):
            perturbation = self.PG(Te_I[i].float().unsqueeze(0).cuda())
            supervisor_code = self.PS(perturbation)
            qB[i, :] = torch.sign(supervisor_code.cpu().data)[0]
        IdB_S = self.SIM.generate_hash_code(Db_I)
        TdB_S = self.STM.generate_hash_code(Db_T)
        P2T_map_PS = CalcMap(qB, TdB_S, Te_L, Db_L, 50)
        P2I_map_PS = CalcMap(qB, IdB_S, Te_L, Db_L, 50)
        print('P2T_map_PS@50: {:.4f}'.format(P2T_map_PS))
        print('P2I_map_PS@50: {:.4f}'.format(P2I_map_PS))
        IdB_A = self.attacked_model.generate_image_hashcode(Db_I)
        TdB_A = self.attacked_model.generate_text_hashcode(Db_T)
        P2T_map_PA = CalcMap(qB, TdB_A, Te_L, Db_L, 50)
        P2I_map_PA = CalcMap(qB, IdB_A, Te_L, Db_L, 50)
        print('P2T_map_PA@50: {:.4f}'.format(P2T_map_PA))
        print('P2I_map_PA@50: {:.4f}'.format(P2I_map_PA))

    def train_attack_model(self, Tr_I, Tr_T, Tr_L):
        print('train attack model...')
        query_sampling_numbers = 2000
        near_sample_number = 5
        index_FS_I = np.random.choice(range(Tr_I.size(0)), query_sampling_numbers, replace = False)
        QI = Tr_I[index_FS_I].type(torch.float).cuda()
        QIB = self.attacked_model.generate_image_hashcode(QI)
        TB = self.attacked_model.generate_text_hashcode(Tr_T)
        index_matrix_after_IT = return_results(index_FS_I, QIB, TB, near_sample_number, 0)

        self.load_surrogate_model()
        self.load_perturbation_model()
        optimizer_IG = torch.optim.Adam(self.IG.parameters(), lr=self.args.alr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.args.alr, betas=(0.5, 0.999))
        criterion_l2 = torch.nn.MSELoss()
        for epoch in range(self.args.ae):
            index = np.random.permutation(query_sampling_numbers)
            for i in range(query_sampling_numbers // self.args.abz + 1):
                end_index = min((i+1)*self.args.abz, query_sampling_numbers)
                num_index = end_index - i*self.args.abz
                ind = index[i*self.args.abz : end_index]
                batch_image = image_normalization(Tr_I[index_matrix_after_IT[ind, 0].numpy()].type(torch.float).cuda())
                batch_perturbation = self.PG(batch_image)
                supervisor_code = self.PS(batch_perturbation)
                batch_adv_image = self.IG(batch_image, batch_perturbation)
                # update D
                if i % 3 ==0:
                    self.set_requires_grad(self.D, True)
                    optimizer_D.zero_grad()
                    batch_image_D = self.D(batch_image)
                    batch_adv_image_D = self.D(batch_adv_image.detach())
                    real_D_loss = self.criterionGAN(batch_image_D, True)
                    adv_D_loss = self.criterionGAN(batch_adv_image_D, False)
                    D_loss = (real_D_loss + adv_D_loss) / 2
                    D_loss.backward()
                    optimizer_D.step()
                # update G
                self.set_requires_grad(self.D, False)
                optimizer_IG.zero_grad()
                # I2T_alienation_loss
                alienation_loss = torch.nn.MarginRankingLoss(margin=0.8)
                I2T_alienation_loss = .0
                for j in range(near_sample_number):
                    batch_adv_image_code = self.SIM(batch_adv_image)
                    batch_text_code = self.STM(Tr_T[index_matrix_after_IT[ind, 1+j].numpy()].type(torch.float).cuda())
                    hamming_dist_IT = calc_hamming(batch_adv_image_code, batch_text_code) / self.args.bit
                    I2T_alienation_loss = I2T_alienation_loss + alienation_loss(hamming_dist_IT.cuda(), torch.zeros(num_index).cuda(), torch.ones(num_index).cuda())
                reconstruction_loss = criterion_l2(batch_adv_image, batch_image)
                logloss = - torch.mean(batch_adv_image_code * supervisor_code) + 1
                batch_adv_image_D = self.D(batch_adv_image)
                adversarial_loss = self.criterionGAN(batch_adv_image_D, True)
                G_loss = 1 * I2T_alienation_loss + 20 * reconstruction_loss + 1 * adversarial_loss # + 5 * logloss # logloss: extra loss
                G_loss.backward()
                optimizer_IG.step()
            print('epoch:{:2d}   D_loss:{:.4f}  real_D_loss:{:.4f}  adv_D_loss:{:.4f}  G_loss:{:.4f}  alienation_loss:{:.4f}  adversarial_loss:{:.4f}  reconstruction_loss:{:.4f}  logloss:{:.4f}'
                .format(epoch, D_loss, real_D_loss, adv_D_loss, G_loss, I2T_alienation_loss/near_sample_number, adversarial_loss, reconstruction_loss, logloss))
        self.save_attack_model()
        print('train attack model done.')

    def test_attack_model(self, Te_I, Te_T, Te_L, Db_I, Db_T, Db_L):
        print('test attack model...')
        self.load_surrogate_model()
        self.load_perturbation_model()
        self.load_attack_model()
        qB_C = torch.zeros([self.Dcfg.query_size, self.args.bit])
        qB_S = torch.zeros([self.Dcfg.query_size, self.args.bit])
        qB_A = torch.zeros([self.Dcfg.query_size, self.args.bit])
        perceptibility = 0.0
        for i in range(self.Dcfg.query_size):
            image = image_normalization(Te_I[i].float().cuda())
            perturbation = self.PG(image.unsqueeze(0))
            adv_image = self.IG(image.unsqueeze(0), perturbation)
            adv_image_pixel = image_restoration(adv_image)
            adv_image_code_S = self.SIM(adv_image_pixel)
            adv_image_code_C = self.attacked_model.image_model(image_restoration(image.unsqueeze(0)))
            adv_image_code_A = self.attacked_model.image_model(adv_image_pixel)
            qB_S[i, :] = torch.sign(adv_image_code_S.cpu().data)[0]
            qB_C[i, :] = torch.sign(adv_image_code_C.cpu().data)
            qB_A[i, :] = torch.sign(adv_image_code_A.cpu().data)
            perceptibility += F.mse_loss((image+1)/2, (adv_image[0]+1)/2).data
        IdB_S = self.SIM.generate_hash_code(Db_I)
        TdB_S = self.STM.generate_hash_code(Db_T)
        IdB_A = self.attacked_model.generate_image_hashcode(Db_I)
        TdB_A = self.attacked_model.generate_text_hashcode(Db_T)
        I2I_map_CC = CalcMap(qB_C, IdB_A, Te_L, Db_L, 50)
        I2T_map_CC = CalcMap(qB_C, TdB_A, Te_L, Db_L, 50)
        I2I_map_SS = CalcMap(qB_S, IdB_S, Te_L, Db_L, 50)
        I2T_map_SS = CalcMap(qB_S, TdB_S, Te_L, Db_L, 50)
        I2I_map_AA = CalcMap(qB_A, IdB_A, Te_L, Db_L, 50)
        I2T_map_AA = CalcMap(qB_A, TdB_A, Te_L, Db_L, 50)
        print('perceptibility: {:.6f}'.format(torch.sqrt(perceptibility/self.Dcfg.query_size)))
        print('I2T_map_CC@50: {:.4f}'.format(I2T_map_CC))
        print('I2I_map_CC@50: {:.4f}'.format(I2I_map_CC))
        print('I2T_map_SS@50: {:.4f}'.format(I2T_map_SS))
        print('I2I_map_SS@50: {:.4f}'.format(I2I_map_SS))
        print('I2T_map_AA@50: {:.4f}'.format(I2T_map_AA))
        print('I2I_map_AA@50: {:.4f}'.format(I2I_map_AA))