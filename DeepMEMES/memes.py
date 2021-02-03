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

from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
from torch.utils.data import TensorDataset, DataLoader



all_scores = {}
def scoring_function(smile,index):
    ## add your scoring function here
    ## if want to search for molecules while optimize negative value multiply by -1 before returning

    score = np.random.randint(0,1000)
    all_scores[index] = score
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
parser.add_argument('--batch_size',default='1024')
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
with open(smiles_path,'r') as f:
    smiles = f.readlines()
    smiles = [i.strip('\n') for i in smiles]

## making inital dataset 
X_init = features[X_index]
Y_init = []
for index in X_index:
    score = scoring_function(smiles[index],index)
    Y_init.append(score)
Y_init = np.array(Y_init)

with open(directory_path+'/start_smiles.txt','w') as f:
    for idx in X_index:
        f.write(smiles[idx]+','+str(all_scores[idx])+'\n')
        smiles_set.add(smiles[idx])
    f.close()

# Initialize samples
X_sample = X_init
Y_sample = Y_init

# Number of iterations
n_iter = iters

print ("starting with {} samples".format(X_sample.shape))
BATCH_SIZE = int(args.batch_size)


class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())


    def forward(self, x):
        mean_x = self.mean_module(x) # self.linear_layer(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

class DeepGPModel(DeepGP):
    def __init__(self, train_x_shape):

        hidden_layer = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=20,
            mean_type='linear',
        )

        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

class GP:
    def __init__(self, train_x, train_y, eps=0.05):
        self.train_x = train_x
        self.train_y = train_y
        self.eps = eps
        self.deepGP = DeepGPModel(train_x.shape)

    def train_gp(self,train_x, train_y):
        train_x = train_x.to(device)    
        train_y = train_y.to(device)
        model = self.deepGP
        model = model.to(device)
        

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.01)

        # "Loss" for GPs - the marginal log likelihood
        mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))
        training_iter = 500
        # training_iter = 1 ## testing
        num_samples = 10 #hyperparamter
        pbar = tqdm.tqdm(range(training_iter))
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        if train_x.shape[0] <= 5000:
            train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)
        prev_best_loss = 1e5
        early_stopping = 0
        for i in pbar:
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
            loss_arr = []
            for x_batch, y_batch in minibatch_iter:
                with gpytorch.settings.num_likelihood_samples(num_samples):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = model(x_batch)
                    # Calc loss and backprop gradients
                    loss = -mll(output, y_batch)
                    loss.backward()
                    
                    loss_arr.append(loss.item())
                    optimizer.step()
            pbar.set_description('Iter %d/%d - Loss: %.3f ' % (
                i + 1, training_iter, np.mean(loss_arr),
            ))

            if np.mean(loss_arr) < prev_best_loss:
                prev_best_loss = np.mean(loss_arr)
                early_stopping = 0
            else:
                early_stopping += 1

            if early_stopping >= 10:
                break



    def compute_ei(self,id):
        model = self.deepGP
        model.eval()
        means = np.array([])
        stds = np.array([])
        for i in tqdm.tqdm(range(0,len(features),BATCH_SIZE*4)):
            test_x = features[i:i+BATCH_SIZE*4]
            test_x = torch.FloatTensor(test_x).to(device)
            with torch.no_grad():
                observed_pred = model.likelihood(model(test_x))
                m = observed_pred.mean
                s = observed_pred.stddev
            m = m.mean(0).detach().cpu().numpy()
            s = s.mean(0).detach().cpu().numpy()
            means = np.append(means,m)
            stds = np.append(stds,s)
    
        imp = means - max(Y_sample) - self.eps
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
    train_x, train_y = torch.FloatTensor(X_sample), torch.FloatTensor(Y_sample)
    gp = GP(train_x, train_y,eps=eps)
    gp.train_gp(train_x, train_y)
    print("Fit Done in :",time.time() - start_time)
    sys.stdout.flush()

    print("Calculatin EI")
    sys.stdout.flush()
    start_time = time.time()
    eis = gp.compute_ei(i)
    print("Calculated EI in:", time.time() - start_time)
    sys.stdout.flush()
    next_indexes = eis.argsort()
    X_next = []
    Y_next = []
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
            score = scoring_function(smiles[index],index)
            Y_next.append(score)
        if count == periter:
            break
    if(len(X_next)==0):
        print("break")
        break
    X_next = np.vstack(X_next)
        
    with open(directory_path+'/iter_'+str(i)+'.txt','w') as f:
        for index in indices:
            if smiles[index] not in smiles_set:
                f.write(smiles[index] + ',' + str(all_scores[index]) + '\n')
    for index in indices:
        smiles_set.add(smiles[index])
    
    print("Iter "+ str(i) + " done")
    sys.stdout.flush()
    
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.append(Y_sample, np.array(Y_next))
    if(len(Y_sample)>=capital):
    	print("capital reached")
    	break
    print(Y_sample.shape)
    sys.stdout.flush()


# In[ ]:




