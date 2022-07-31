#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm


import pickle
import sys
import argparse
import os

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import pickle
import time
import tqdm
import ipdb


all_scores_1 = {}
all_scores_2 = {}
def scoring_function_1(smile,index):
    ## add your scoring function here
    ## if want to search for molecules while optimize negative value multiply by -1 before returning

    score = np.random.randint(0,1000)
    all_scores_1[index] = score
    ## optimizing for -ve value
    return -1*score
    ## optimizing for +ve value
    return +1*score

def scoring_function_2(smile,index):
    ## add your scoring function here
    ## if want to search for molecules while optimize negative value multiply by -1 before returning

    score = np.random.randint(0,1000)
    all_scores_2[index] = score
    ## optimizing for -ve value
    return -1*score
    ## optimizing for +ve value
    return +1*score


parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--rec', required=True)
parser.add_argument('--cuda',required=True)
parser.add_argument('--feature', default='mol2vec')
parser.add_argument('--features_path', default='features.pickle')
parser.add_argument('--smiles_path', default='all.txt')
parser.add_argument('--iters',default='40')
parser.add_argument('--capital',default='15000')
parser.add_argument('--initial',default='5000')
parser.add_argument('--periter',default='500')
parser.add_argument('--n_cluster',default='20')
parser.add_argument('--save_eis', required=False,default="False")
parser.add_argument('--eps',default='0.05')
args = parser.parse_args()
run = int(args.run)
iters = int(args.iters)
capital = int(args.capital)
initial = int(args.initial)
periter = int(args.periter)
n_cluster = int(args.n_cluster)
eps = float(args.eps)
rec = args.rec
feat = args.feature
features_path = args.features_path
device = args.cuda
if device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device('cuda:{}'.format(device))
save_eis = args.save_eis == "True"
print("Using device: ", device)

directory_path = './gp_runs/{}-'.format(feat)+rec+'/run'+str(run)

try:
    os.makedirs(directory_path)
except:
    pass

with open(directory_path+'/config.txt','w') as f:
	f.write("periter:	"+str(periter)+"\n")
	f.write("initial:	"+str(initial)+"\n")
	f.write("feature:	"+str(feat)+"\n")
	f.write("eps:	"+str(eps)+"\n")
	f.write("rec:	"+str(rec)+"\n")
	f.close()

pickle_obj = pickle.load(open(features_path,"rb"))
features = np.array(pickle_obj["data"])

features = np.nan_to_num(features)

features = features - features.min()
features = 2 * (features / features.max()) - 1
print(features.shape)


## loading cluster labels
labels = np.loadtxt("labels{}.txt".format(n_cluster))


## selecting inital points
X_index = []
for i in range(n_cluster):
    X_index.extend(np.random.choice(np.where(labels==i)[0],int(initial//n_cluster)))
X_index = np.array(X_index)

## loading all smiles from complete dataset
smiles_set = set()
with open(args.smiles_path,'r') as f:
    smiles = f.readlines()
    smiles = [i.strip('\n') for i in smiles]

## making inital dataset 
X_init = features[X_index]
Y_init_1 = []
Y_init_2 = []
for index in X_index:
    score = scoring_function_1(smiles[index],index)
    Y_init_1.append(score)

for index in X_index:
    score = scoring_function_2(smiles[index],index)
    Y_init_2.append(score)
Y_init_1 = np.array(Y_init_1)
Y_init_2 = np.array(Y_init_2)

with open(directory_path+'/start_smiles.txt','w') as f:
    for idx in X_index:
        f.write(smiles[idx]+','+str(all_scores[idx])+'\n')
        smiles_set.add(smiles[idx])
    f.close()

# Initialize samples
X_sample = X_init
Y_sample_1 = Y_init_1
Y_sample_2 = Y_init_2

# Number of iterations
n_iter = iters

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GP:
    def __init__(self, train_x, train_y, eps=0.05):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(train_x, train_y, self.likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.eps = eps

    def train_gp(self,train_x, train_y):
        train_x = train_x.to(device)    
        train_y = train_y.to(device)
        model = self.model.to(device)
        likelihood = self.likelihood.to(device)
        
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        training_iter = 500
        pbar = tqdm.tqdm(range(training_iter))
        prev_best_loss = 1e5
        early_stopping = 0
        for i in pbar:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            
            pbar.set_description('Iter %d/%d - Loss: %.3f ' % (
                i + 1, training_iter, loss.item(),
            ))
            if loss.item() < prev_best_loss:
                prev_best_loss = loss.item()
                early_stopping = 0
            else:
                early_stopping += 1

            if early_stopping >= 10:
                break
            optimizer.step()

    def compute_ei(self,id,best_val):
        self.model.eval()
        self.likelihood.eval()
        means = np.array([])
        stds = np.array([])
        #20000 is system dependent. Change according to space in GPU
        eval_bs_size = 20000
        for i in tqdm.tqdm(range(0,len(features),eval_bs_size)):
            test_x = features[i:i+eval_bs_size]
            test_x = torch.FloatTensor(test_x).to(device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(test_x))
                m = observed_pred.mean
                s = observed_pred.stddev
            m = m.cpu().numpy()
            s = s.cpu().numpy()
            means = np.append(means,m)
            stds = np.append(stds,s)

        imp = means - best_val - self.eps
        Z = imp/stds
        eis = imp * norm.cdf(Z) + stds * norm.pdf(Z)
        eis[stds == 0.0] = 0.0
        if save_eis:
            np.savetxt(directory_path+'/eis_' + str(id)+'.out',eis)
        return eis




## Iterative algorithm
for i in range(iters):
    print("Fit Start")
    sys.stdout.flush()
    # initialize likelihood and model
    start_time = time.time()

    ## property 1
    print("Running for property 1")
    train_x, train_y = torch.FloatTensor(X_sample), torch.FloatTensor(Y_sample_1)
    gp = GP(train_x, train_y,eps=eps)
    gp.train_gp(train_x, train_y)
    print("Fit Done in :",time.time() - start_time)
    sys.stdout.flush()

    print("Calculatin EI")
    sys.stdout.flush()
    start_time = time.time()
    eis_1 = gp.compute_ei(i,max(Y_sample_1))

    ## property 2
    print("Running for property 2")
    train_x, train_y = torch.FloatTensor(X_sample), torch.FloatTensor(Y_sample_2)
    gp = GP(train_x, train_y,eps=eps)
    gp.train_gp(train_x, train_y)
    print("Fit Done in :",time.time() - start_time)
    sys.stdout.flush()

    print("Calculatin EI")
    sys.stdout.flush()
    start_time = time.time()
    eis_2 = gp.compute_ei(i,max(Y_sample_2))

    eis = eis_1*eis_2

    next_indexes = eis.argsort()
    X_next = []
    Y_next_1 = []
    Y_next_2 = []
    count = 0
    indices = []
    if len(X_sample) < 1000:
        periter = 200
    else:
        periter = int(args.periter)
    for index in next_indexes[::-1]:
        if smiles[index] in smiles_set:
            continue
        else:
            count+=1
            indices.append(index)
            X_next.append(features[index])
            score = scoring_function_1(smiles[index],index)
            Y_next_1.append(score)
            score = scoring_function_2(smiles[index],index)
            Y_next_2.append(score)


        if count == periter:
            break
    if(len(X_next)==0):
        print("break")
        break
    X_next = np.vstack(X_next)
        
    with open(directory_path+'/iter_'+str(i)+'.txt','w') as f:
        for index in indices:
            if smiles[index] not in smiles_set:
                f.write(smiles[index] + ',' + str(all_scores_1[index]) + ',' + str(all_scores_2[index]) + '\n')
    for index in indices:
        smiles_set.add(smiles[index])
    
    print("Iter "+ str(i) + " done")
    sys.stdout.flush()
    
    X_sample = np.vstack((X_sample, X_next))
    Y_sample_1 = np.append(Y_sample_1, np.array(Y_next_1))
    Y_sample_2 = np.append(Y_sample_2, np.array(Y_next_2))
    if(len(Y_sample_2)>=capital):
    	print("capital reached")
    	break
    print(Y_sample_2.shape)
    sys.stdout.flush()


# In[ ]:




