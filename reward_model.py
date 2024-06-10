import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time

from scipy.stats import norm
from multiprocessing import Queue
import pandas as pd

device = 'cuda'

# add_data_batch

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 teacher_num_ratings=2):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        self.num_ratings = teacher_num_ratings
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin


        self.num_timesteps = 0
        self.member_1_pred_reward = []
        self.member_2_pred_reward = []
        self.member_3_pred_reward = []
        self.real_rewards = []
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
    
    def get_mean_and_std(self, x_1):
        probs = []
        rewards = []
        for member in range(self.de):
            with torch.no_grad():
                r_hat = self.r_hat_member(x_1, member=member)
                r_hat = r_hat.sum(axis=1)
                rewards.append(r_hat.cpu().numpy())

        rewards = np.array(rewards)

        return np.mean(rewards, axis=0).flatten(), np.std(rewards, axis=0).flatten()
    
    def get_entropy(self, x_1):
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    
    def p_hat_entropy(self, x_1, member=-1):
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat = r_hat1
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent
    
    def r_hat_member(self, x, member=-1):
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat = r_hat1                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=100):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1 = None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] 
        r_t_1 = train_targets[batch_index_1] 
                
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) 
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) 

        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) 
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)
                
        return sa_t_1, r_t_1

    def put_queries(self, sa_t_1, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
            
    def get_label(self, sa_t_1, r_t_1):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            r_t_1 = r_t_1[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)

        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
        temp_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_1 = np.zeros_like(temp_r_t_1)

        rewards = np.sum(r_t_1, axis=1)
        
        if self.num_ratings == 2:
            sum_r_t_1[(temp_r_t_1 < 25)] = 0
            sum_r_t_1[(temp_r_t_1 >= 25.0001)] = 1
        elif self.num_ratings == 3:
            sum_r_t_1[(temp_r_t_1 < 16.67)] = 0
            sum_r_t_1[(temp_r_t_1 >= 16.6701) & (temp_r_t_1 < 33.33)] = 1
            sum_r_t_1[(temp_r_t_1 >= 33.3301)] = 2
        elif self.num_ratings == 4:
            sum_r_t_1[(temp_r_t_1 < 12.5)] = 0
            sum_r_t_1[(temp_r_t_1 >= 12.5001) & (temp_r_t_1 < 25)] = 1
            sum_r_t_1[(temp_r_t_1 >= 25.0001) & (temp_r_t_1 < 37.5)] = 2
            sum_r_t_1[(temp_r_t_1 >= 37.5001)] = 3
        elif self.num_ratings == 5:
            sum_r_t_1[(temp_r_t_1 < 10)] = 0
            sum_r_t_1[(temp_r_t_1 >= 10.0001) & (temp_r_t_1 < 20)] = 1
            sum_r_t_1[(temp_r_t_1 >= 20.0001) & (temp_r_t_1 < 30)] = 2
            sum_r_t_1[(temp_r_t_1 >= 30.0001) & (temp_r_t_1 < 40)] = 3
            sum_r_t_1[(temp_r_t_1 >= 40.0001)] = 4
        else:
            sum_r_t_1[temp_r_t_1 < 8.3] = 0
            sum_r_t_1[(temp_r_t_1 >= 8.3001) & (temp_r_t_1 < 16.67)] = 1
            sum_r_t_1[(temp_r_t_1 >= 16.6701) & (temp_r_t_1 < 25)] = 2
            sum_r_t_1[(temp_r_t_1 >= 25.0001) & (temp_r_t_1 < 33.33)] = 3
            sum_r_t_1[(temp_r_t_1 >= 33.3301) & (temp_r_t_1 < 41.66)] = 4
            sum_r_t_1[(temp_r_t_1 >= 41.6601)] = 5
        labels = sum_r_t_1
        
        return sa_t_1, r_t_1, labels
        
    def uniform_sampling(self):
        # get queries
        sa_t_1, r_t_1 =  self.get_queries(
            mb_size=self.mb_size)
            
        # get labels
        sa_t_1, r_t_1, labels = self.get_label(
            sa_t_1, r_t_1)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1,labels)
        
        return len(labels)
    
    def disagreement_sampling(self):
        
        # get queries
        sa_t_1, r_t_1 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_mean_and_std(sa_t_1)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]       
        
        # get labels
        sa_t_1, r_t_1, labels = self.get_label(
            sa_t_1, r_t_1)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, labels)
        
        return len(labels)
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                labels = self.buffer_label[idxs]


                if self.num_ratings == 2:
                    num_B = 0
                    num_G = 0

                    for label in labels:
                        
                        if label == 0:
                            num_B += 1
                        elif label == 1:
                            num_G += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=2)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    
                    upper_bound_B = np_pred[num_B-1]
                    upper_bound_G = np_pred[num_B+num_G-1]

                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)

                    k =30
                    
                    Q_0 = -(pred-0)*(pred-upper_bound_B)*(k)
                    Q_1 = -(pred-upper_bound_B)*(pred-1)*(k)
                    

                    our_Q = torch.cat([Q_0, Q_1], axis=-1)
                    
                    curr_loss = self.CEloss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct
                
                elif self.num_ratings == 3:
                    num_B = 0
                    num_N = 0
                    num_G = 0

                    for label in labels:
                        
                        if label == 0:
                            num_B += 1
                        elif label == 1:
                            num_N += 1
                        elif label == 2:
                            num_G += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=3)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    
                    upper_bound_B = np_pred[num_B-1]
                    upper_bound_N = np_pred[num_B+num_N-1]
                    upper_bound_G = np_pred[num_B+num_N+num_G-1]

                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_N = torch.as_tensor(upper_bound_N).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)

                    k =30
                    
                    Q_0 = -(pred-0)*(pred-upper_bound_B)*(k)
                    Q_1 = -(pred-upper_bound_B)*(pred-upper_bound_N)*(k)
                    Q_2 = -(pred-upper_bound_N)*(pred-1)*(k)
                    

                    our_Q = torch.cat([Q_0, Q_1, Q_2], axis=-1)
                    
                    curr_loss = self.CEloss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct
                
                elif self.num_ratings == 4:
                    num_VB = 0
                    num_B = 0
                    num_G = 0
                    num_VG = 0

                    for label in labels:
                        if label == 0:
                            num_VB += 1
                        elif label == 1:
                            num_B += 1
                        elif label == 2:
                            num_G += 1
                        elif label == 3:
                            num_VG += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=4)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    upper_bound_VB = np_pred[num_VB-1]
                    upper_bound_B = np_pred[num_VB+num_B-1]
                    upper_bound_G = np_pred[num_VB+num_B+num_G-1]
                    upper_bound_VG = np_pred[num_VB+num_B+num_G+num_VG-1]

                    upper_bound_VB = torch.as_tensor(upper_bound_VB).to(device)
                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)
                    upper_bound_VG = torch.as_tensor(upper_bound_VG).to(device)

                    k =30
                    Q_0 = -(pred-0)*(pred-upper_bound_VB)*(k)
                    Q_1 = -(pred-upper_bound_VB)*(pred-upper_bound_B)*(k)
                    Q_2 = -(pred-upper_bound_B)*(pred-upper_bound_G)*(k)
                    Q_3 = -(pred-upper_bound_G)*(pred-1)*(k)

                    our_Q = torch.cat([Q_0, Q_1, Q_2, Q_3], axis=-1)
                    
                    curr_loss = self.CEloss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct

                elif self.num_ratings == 5:
                    num_VB = 0
                    num_B = 0
                    num_N = 0
                    num_G = 0
                    num_VG = 0

                    for label in labels:
                        if label == 0:
                            num_VB += 1
                        elif label == 1:
                            num_B += 1
                        elif label == 2:
                            num_N += 1
                        elif label == 3:
                            num_G += 1
                        elif label == 4:
                            num_VG += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=5)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    upper_bound_VB = np_pred[num_VB-1]
                    upper_bound_B = np_pred[num_VB+num_B-1]
                    upper_bound_N = np_pred[num_VB+num_B+num_N-1]
                    upper_bound_G = np_pred[num_VB+num_B+num_N+num_G-1]
                    upper_bound_VG = np_pred[num_VB+num_B+num_N+num_G+num_VG-1]

                    upper_bound_VB = torch.as_tensor(upper_bound_VB).to(device)
                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_N = torch.as_tensor(upper_bound_N).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)
                    upper_bound_VG = torch.as_tensor(upper_bound_VG).to(device)

                    k =30
                    Q_0 = -(pred-0)*(pred-upper_bound_VB)*(k)
                    Q_1 = -(pred-upper_bound_VB)*(pred-upper_bound_B)*(k)
                    Q_2 = -(pred-upper_bound_B)*(pred-upper_bound_N)*(k)
                    Q_3 = -(pred-upper_bound_N)*(pred-upper_bound_G)*(k)
                    Q_4 = -(pred-upper_bound_G)*(pred-1)*(k)

                    our_Q = torch.cat([Q_0, Q_1, Q_2, Q_3, Q_4], axis=-1)
                    
                    curr_loss = self.CEloss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct
                
                elif self.num_ratings == 6:
                    num_VB = 0
                    num_B = 0
                    num_N = 0
                    num_G = 0
                    num_VG = 0
                    num_P = 0

                    for label in labels:
                        if label == 0:
                            num_VB += 1
                        elif label == 1:
                            num_B += 1
                        elif label == 2:
                            num_N += 1
                        elif label == 3:
                            num_G += 1
                        elif label == 4:
                            num_VG += 1
                        elif label == 5:
                            num_P += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=6)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    upper_bound_VB = np_pred[num_VB-1]
                    upper_bound_B = np_pred[num_VB+num_B-1]
                    upper_bound_N = np_pred[num_VB+num_B+num_N-1]
                    upper_bound_G = np_pred[num_VB+num_B+num_N+num_G-1]
                    upper_bound_VG = np_pred[num_VB+num_B+num_N+num_G+num_VG-1]
                    upper_bound_P = np_pred[num_VB+num_B+num_N+num_G+num_VG+num_P-1]

                    upper_bound_VB = torch.as_tensor(upper_bound_VB).to(device)
                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_N = torch.as_tensor(upper_bound_N).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)
                    upper_bound_VG = torch.as_tensor(upper_bound_VG).to(device)
                    upper_bound_P = torch.as_tensor(upper_bound_P).to(device)

                    k =30
                    Q_0 = -(pred-0)*(pred-upper_bound_VB)*(k)
                    Q_1 = -(pred-upper_bound_VB)*(pred-upper_bound_B)*(k)
                    Q_2 = -(pred-upper_bound_B)*(pred-upper_bound_N)*(k)
                    Q_3 = -(pred-upper_bound_N)*(pred-upper_bound_G)*(k)
                    Q_4 = -(pred-upper_bound_G)*(pred-upper_bound_VG)*(k)
                    Q_5 = -(pred-upper_bound_VG) * (pred-1)*(k)

                    our_Q = torch.cat([Q_0, Q_1, Q_2, Q_3, Q_4, Q_5], axis=-1)
                    
                    curr_loss = self.CEloss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct

                
            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc

    
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                labels = self.buffer_label[idxs]

                if self.num_ratings == 2:
                    num_B = 0
                    num_G = 0

                    for label in labels:
                        
                        if label == 0:
                            num_B += 1
                        elif label == 1:
                            num_G += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=2)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    
                    upper_bound_B = np_pred[num_B-1]
                    upper_bound_G = np_pred[num_B+num_G-1]

                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)

                    k =30
                    
                    Q_0 = -(pred-0)*(pred-upper_bound_B)*(k)
                    Q_1 = -(pred-upper_bound_B)*(pred-1)*(k)
                    

                    our_Q = torch.cat([Q_0, Q_1], axis=-1)
                    
                    curr_loss = self.softXEnt_loss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct
                
                elif self.num_ratings == 3:
                    num_B = 0
                    num_N = 0
                    num_G = 0

                    for label in labels:
                        
                        if label == 0:
                            num_B += 1
                        elif label == 1:
                            num_N += 1
                        elif label == 2:
                            num_G += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=3)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    
                    upper_bound_B = np_pred[num_B-1]
                    upper_bound_N = np_pred[num_B+num_N-1]
                    upper_bound_G = np_pred[num_B+num_N+num_G-1]

                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_N = torch.as_tensor(upper_bound_N).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)

                    k =30
                    
                    Q_0 = -(pred-0)*(pred-upper_bound_B)*(k)
                    Q_1 = -(pred-upper_bound_B)*(pred-upper_bound_N)*(k)
                    Q_2 = -(pred-upper_bound_N)*(pred-1)*(k)
                    

                    our_Q = torch.cat([Q_0, Q_1, Q_2], axis=-1)
                    
                    curr_loss = self.softXEnt_loss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct
                
                elif self.num_ratings == 4:
                    num_VB = 0
                    num_B = 0
                    num_G = 0
                    num_VG = 0

                    for label in labels:
                        if label == 0:
                            num_VB += 1
                        elif label == 1:
                            num_B += 1
                        elif label == 2:
                            num_G += 1
                        elif label == 3:
                            num_VG += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=4)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    upper_bound_VB = np_pred[num_VB-1]
                    upper_bound_B = np_pred[num_VB+num_B-1]
                    upper_bound_G = np_pred[num_VB+num_B+num_G-1]
                    upper_bound_VG = np_pred[num_VB+num_B+num_G+num_VG-1]

                    upper_bound_VB = torch.as_tensor(upper_bound_VB).to(device)
                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)
                    upper_bound_VG = torch.as_tensor(upper_bound_VG).to(device)

                    k =30
                    Q_0 = -(pred-0)*(pred-upper_bound_VB)*(k)
                    Q_1 = -(pred-upper_bound_VB)*(pred-upper_bound_B)*(k)
                    Q_2 = -(pred-upper_bound_B)*(pred-upper_bound_G)*(k)
                    Q_3 = -(pred-upper_bound_G)*(pred-1)*(k)

                    our_Q = torch.cat([Q_0, Q_1, Q_2, Q_3], axis=-1)
                    
                    curr_loss = self.softXEnt_loss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct

                elif self.num_ratings == 5:
                    num_VB = 0
                    num_B = 0
                    num_N = 0
                    num_G = 0
                    num_VG = 0

                    for label in labels:
                        if label == 0:
                            num_VB += 1
                        elif label == 1:
                            num_B += 1
                        elif label == 2:
                            num_N += 1
                        elif label == 3:
                            num_G += 1
                        elif label == 4:
                            num_VG += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=5)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    upper_bound_VB = np_pred[num_VB-1]
                    upper_bound_B = np_pred[num_VB+num_B-1]
                    upper_bound_N = np_pred[num_VB+num_B+num_N-1]
                    upper_bound_G = np_pred[num_VB+num_B+num_N+num_G-1]
                    upper_bound_VG = np_pred[num_VB+num_B+num_N+num_G+num_VG-1]

                    upper_bound_VB = torch.as_tensor(upper_bound_VB).to(device)
                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_N = torch.as_tensor(upper_bound_N).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)
                    upper_bound_VG = torch.as_tensor(upper_bound_VG).to(device)

                    k =30
                    Q_0 = -(pred-0)*(pred-upper_bound_VB)*(k)
                    Q_1 = -(pred-upper_bound_VB)*(pred-upper_bound_B)*(k)
                    Q_2 = -(pred-upper_bound_B)*(pred-upper_bound_N)*(k)
                    Q_3 = -(pred-upper_bound_N)*(pred-upper_bound_G)*(k)
                    Q_4 = -(pred-upper_bound_G)*(pred-1)*(k)

                    our_Q = torch.cat([Q_0, Q_1, Q_2, Q_3, Q_4], axis=-1)
                    
                    curr_loss = self.softXEnt_loss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct
                
                elif self.num_ratings == 6:
                    num_VB = 0
                    num_B = 0
                    num_N = 0
                    num_G = 0
                    num_VG = 0
                    num_P = 0

                    for label in labels:
                        if label == 0:
                            num_VB += 1
                        elif label == 1:
                            num_B += 1
                        elif label == 2:
                            num_N += 1
                        elif label == 3:
                            num_G += 1
                        elif label == 4:
                            num_VG += 1
                        elif label == 5:
                            num_P += 1

                    labels = torch.from_numpy(labels.flatten()).long().to(device)
                    target_onehot = F.one_hot(labels, num_classes=6)
                    
                    if member == 0:
                        total += labels.size(0)
                    
                    # get logits
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat = r_hat1
                    
                    pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                    sorted_indices = pred[:, 0].sort()[1]
                    np_pred = pred[sorted_indices]
                    np_pred = np_pred.tolist()
                    
                    upper_bound_VB = np_pred[num_VB-1]
                    upper_bound_B = np_pred[num_VB+num_B-1]
                    upper_bound_N = np_pred[num_VB+num_B+num_N-1]
                    upper_bound_G = np_pred[num_VB+num_B+num_N+num_G-1]
                    upper_bound_VG = np_pred[num_VB+num_B+num_N+num_G+num_VG-1]
                    upper_bound_P = np_pred[num_VB+num_B+num_N+num_G+num_VG+num_P-1]

                    upper_bound_VB = torch.as_tensor(upper_bound_VB).to(device)
                    upper_bound_B = torch.as_tensor(upper_bound_B).to(device)
                    upper_bound_N = torch.as_tensor(upper_bound_N).to(device)
                    upper_bound_G = torch.as_tensor(upper_bound_G).to(device)
                    upper_bound_VG = torch.as_tensor(upper_bound_VG).to(device)
                    upper_bound_P = torch.as_tensor(upper_bound_P).to(device)

                    k =30
                    Q_0 = -(pred-0)*(pred-upper_bound_VB)*(k)
                    Q_1 = -(pred-upper_bound_VB)*(pred-upper_bound_B)*(k)
                    Q_2 = -(pred-upper_bound_B)*(pred-upper_bound_N)*(k)
                    Q_3 = -(pred-upper_bound_N)*(pred-upper_bound_G)*(k)
                    Q_4 = -(pred-upper_bound_G)*(pred-upper_bound_VG)*(k)
                    Q_5 = -(pred-upper_bound_VG) * (pred-1)*(k)

                    our_Q = torch.cat([Q_0, Q_1, Q_2, Q_3, Q_4, Q_5], axis=-1)
                    
                    curr_loss = self.softXEnt_loss(our_Q, target_onehot)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    
                    # compute acc
                    _, predicted = torch.max(our_Q, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct

                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
