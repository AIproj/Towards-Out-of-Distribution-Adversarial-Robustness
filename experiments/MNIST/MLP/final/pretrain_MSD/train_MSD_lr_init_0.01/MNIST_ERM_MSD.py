import numpy as np
import torch
import time
timer = 0

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F

import advertorch.attacks as attacks
from attacks.deepfool import DeepfoolLinfAttack
import torch.nn as nn
from autoattack import AutoAttack

from advertorch.context import ctx_noparamgrad_and_eval
from torch.utils.tensorboard import SummaryWriter

import random

import argparse

argument_parser = argparse.ArgumentParser()

argument_parser.add_argument("--lr_init", type=float, help="Initial learning rate value, default=0.01. CAREFUL: this will be divided by beta, since the ERM term is multiplied by beta in the objective.")

parsed_args = argument_parser.parse_args()


# Make sure validation splits are the same at all time (e.g. even after loading)
seed = 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_workers = 0
# Make sure test_data is a multiple of batch_size_test
batch_size_train_and_valid = 128
batch_size_test = 100

# proportion of full training set used for validation
valid_size = 0.2




transform = transforms.ToTensor()
train_and_valid_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)

num_valid_samples = int(np.floor(valid_size * len(train_and_valid_data)))
num_train_samples = len(train_and_valid_data) - num_valid_samples
train_data, valid_data = torch.utils.data.random_split(train_and_valid_data, [num_train_samples, num_valid_samples], generator=torch.Generator().manual_seed(seed))

train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size_train_and_valid)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size_train_and_valid)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size_test)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        
    def forward(self,x):
        # vectorise input
        x = x.view(-1,28*28)
        # Hidden layer 1 + relu
        x = F.relu(self.fc1(x))
        # Hidden layer 2 + relu
        x = F.relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)
        return x


model = Net()
# model.to(device)


model.load_state_dict(torch.load('model_no_dropout.pt'))
model.to(device)













# divided by 10 eps, eps_iter and CW's lr, added as input binary_search_steps to CW attacks


adversary_PGD_Linf_std = attacks.LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_PGD_L2_std = attacks.L2PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=2.,
    nb_iter=40, eps_iter=0.1, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_PGD_L1_std = attacks.L1PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=10.,
    nb_iter=40, eps_iter=0.5, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_CW = attacks.CarliniWagnerL2Attack(
    model, num_classes=10, max_iterations=20, learning_rate=0.1,
    binary_search_steps=5, clip_min=0.0, clip_max=1.0)

adversary_deepfool = DeepfoolLinfAttack(
        model, num_classes=10, nb_iter=30, eps=0.11, clip_min=0.0, clip_max=1.0)

# Unseen attacks used for validation, has bigger learning rate and number of iterations
adversary_CW_unseen = attacks.CarliniWagnerL2Attack(
    model, num_classes=10, max_iterations=30, learning_rate=0.12,
    binary_search_steps=7, clip_min=0.0, clip_max=1.0)

adversary_PGD_Linf_unseen = attacks.LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.4,
    nb_iter=40, eps_iter=0.033, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_deepfool_unseen = DeepfoolLinfAttack(
        model, num_classes=10, nb_iter=50, eps=0.4, clip_min=0.0, clip_max=1.0)

adversary_autoattack_unseen = AutoAttack(model, norm='Linf', eps=.3, 
        version='standard', seed=None, verbose=False)



def loss_helper(model, data_all_domains, label_all_domains, num_domains, num_correct_per_domain, tensor_list_losses_epoch):
    list_losses = []
    
    for domain in range(0, num_domains):
        preds = model(data_all_domains[domain])
        list_losses.append(F.cross_entropy(preds, label_all_domains[domain]))
        num_correct_per_domain[domain] += ((torch.argmax(preds, dim=1) == label_all_domains[domain]).sum().item())
    
    # Some spaghetti going on here between torch and lists types, as evidenced by how the loss_helper() is called in compute_loss()
    tensor_list_losses = torch.stack(list_losses)
    
    ERM_term = torch.sum(tensor_list_losses) / num_domains
    REx_variance_term = torch.var(tensor_list_losses)
    
    tensor_list_losses_epoch += tensor_list_losses
    
    return ERM_term, REx_variance_term

def REx_loss(ERM_term, REx_variance_term, beta):
    return beta * REx_variance_term + ERM_term

 
def compute_loss(is_REx, beta, loss_terms, model, list_data_all_domains, list_label_all_domains, num_domains, 
                 num_train_correct_preds_per_domain, tensor_list_losses_epoch_train):
    if is_REx:
        ERM_term, REx_variance_term = loss_helper(model, list_data_all_domains, list_label_all_domains, num_domains, num_train_correct_preds_per_domain, tensor_list_losses_epoch_train)
        loss_terms_temp = [ERM_term.item(), REx_variance_term.item()]
        loss_terms += np.array(loss_terms_temp)
        loss = REx_loss(ERM_term, REx_variance_term, beta)
    else:
        ERM_term, _ = loss_helper(model, list_data_all_domains, list_label_all_domains, num_domains, num_train_correct_preds_per_domain, tensor_list_losses_epoch_train)
        loss_terms += np.array([ERM_term.item()])
        loss = ERM_term
    return loss









### The following functions are taken from "Adversarial Robustness Against the Union of Multiple Perturbation Models"
### See https://github.com/locuslab/robust_union/
def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None].squeeze(1).squeeze(1).squeeze(1)

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l0_squeezed(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float().squeeze(1).squeeze(1).squeeze(1)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def proj_simplex(device, v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).to(device)
    comp = (vec > (cssv - s))

    u = comp.cumsum(dim = 2)
    w = (~comp).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.Tensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w

def l1_dir_topk(grad, delta, X, gap, k = 20) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
#     neg1 = (grad < 0)*(X_curr == 0)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
#     neg2 = (grad > 0)*(X_curr == 1)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval) * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)

def proj_l1ball(device, x, epsilon=10):
    assert epsilon > 0
#     ipdb.set_trace()
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
#         y = x* epsilon/norms_l1(x)
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(device, u, s=epsilon)
    # compute the solution to the original problem on v
    y = y.view(-1,1,28,28)
    y *= x.sign()
    return y

def msd_v0(device, model, X,y, epsilon_l_inf = 0.3, epsilon_l_2= 2.0, epsilon_l_1 = 10, alpha_l_inf = 0.01, alpha_l_2 = 0.1, alpha_l_1 = 0.5, num_iter = 40):
    delta = torch.zeros_like(X,requires_grad = True)
    max_delta = torch.zeros_like(X)
    max_max_delta = torch.zeros_like(X)
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_max_loss = torch.zeros(y.shape[0]).to(y.device)
    alpha_l_1_default = alpha_l_1
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        with torch.no_grad():                
            #For L_2
            delta_l_2  = delta.data + alpha_l_2*delta.grad / norms(delta.grad)      
            delta_l_2 *= epsilon_l_2 / norms(delta_l_2).clamp(min=epsilon_l_2)
            delta_l_2  = torch.min(torch.max(delta_l_2, -X), 1-X) # clip X+delta to [0,1]

            #For L_inf
            delta_l_inf=  (delta.data + alpha_l_inf*delta.grad.sign()).clamp(-epsilon_l_inf,epsilon_l_inf)
            delta_l_inf = torch.min(torch.max(delta_l_inf, -X), 1-X) # clip X+delta to [0,1]

            #For L1
            k = random.randint(5,20)
            alpha_l_1 = (alpha_l_1_default/k)*20
            delta_l_1  = delta.data + alpha_l_1*l1_dir_topk(delta.grad, delta.data, X, alpha_l_1, k = k)
            delta_l_1 = proj_l1ball(device, delta_l_1, epsilon_l_1)
            delta_l_1  = torch.min(torch.max(delta_l_1, -X), 1-X) # clip X+delta to [0,1]
            
            #Compare
            delta_tup = (delta_l_1, delta_l_2, delta_l_inf)
            max_loss = torch.zeros(y.shape[0]).to(y.device)
            for delta_temp in delta_tup:
                loss_temp = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_temp), y)
                max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                max_loss = torch.max(max_loss, loss_temp)
            delta.data = max_delta.data
            max_max_delta[max_loss> max_max_loss] = max_delta[max_loss> max_max_loss]
            max_max_loss[max_loss> max_max_loss] = max_loss[max_loss> max_max_loss]
        delta.grad.zero_()

    return max_max_delta
















    





waterfall = False
is_REx = False
resume = False
use_unseen_attacks = False


# number of epochs to train the model
n_epochs_AIT = 10001
# initialize tracker for minimum validation loss
valid_loss_min = np.Inf  # set initial "min" to infinity
beta = 0.001
beta_prev = beta
lr_init = 0.1 # / beta
best_epoch = 0
starting_epoch = 0
# the following var decides when we start waterfalling (includes the chosen epoch)
waterfall_epoch = 326
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr = lr_init, momentum=momentum)



if resume:
    checkpoint = torch.load("experiments/CIFAR10/ResNet18/pretrain_hard_PGD/ERM_lr_init_0.01_beta_on_var/model_AIT_ERM_330.pt")
    starting_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['current_model'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimiser'])
    lr_init = checkpoint['lr_init']
    try:
        beta = checkpoint['beta']
        beta_prev = checkpoint['beta_prev']
    except:
        print("Beta and beta_prev loaded from default of resp ", beta, " and ", beta_prev) 
    best_epoch = checkpoint['best_epoch']



# MODIFY GIVEN PARSED ARGUMENTS
try:
    if parsed_args.lr_init:
        lr_init = parsed_args.lr_init # / beta
        print("lr_init set to %f " % parsed_args.lr_init)
        if resume:
            print("WARNING: an argument was passed to override the lr_init loaded from the checkpoint !")
        
except:
    print("No learning rate passed as argument; using default value of ", lr_init)
        

TRAINED_MODEL_PATH = "experiments/MNIST/MLP/pretrain_MSD/train_MSD"
# DON'T FORGET TO REMOVE
# DON'T FORGET TO REMOVE
# DON'T FORGET TO REMOVE
# DON'T FORGET TO REMOVE
# DON'T FORGET TO REMOVE
# DON'T FORGET TO REMOVE
# DON'T FORGET TO REMOVE# DON'T FORGET TO REMOVE
# DON'T FORGET TO REMOVE
TRAINED_MODEL_PATH += "_lr_init_" + str(lr_init) #+ "_cont"
TRAINED_MODEL_PATH += "/"
path_of_checkpoint = ""
writer = SummaryWriter(TRAINED_MODEL_PATH)




for epoch in range(starting_epoch, n_epochs_AIT):
# #     if (epoch > 0) and (epoch % 10 == 0) and (epoch < 40):
# # #        print("Learning rate halved at epoch ", epoch)
# #         lr_init = lr_init * 0.5
    if epoch % 5 == 0:
        use_unseen_attacks = True
    else:
        use_unseen_attacks = False


    if waterfall:
        beta_prev = beta
        if epoch == waterfall_epoch:
            beta = 10
#             lr_init = lr_init * beta_prev / beta
#         if (epoch > waterfall_epoch) and (epoch % 10 == 0):
#             beta = beta * 0.5
# #            lr_init = lr_init * beta_prev / beta
    
    #TODO fix the fact that with waterfall and momentum, when the loss and its gradient are rescaled,
    # momentum uses gradients of different scales (this assumes the ERM term dominates in the case of REx)
    #TODO CAREFUL, IF CHANGES ARE MADE TO BETA'S BEHAVIOUR
    momentum = 0.9 # * beta / beta_prev
    optimizer.param_groups[0]['momentum'] = momentum
    optimizer.param_groups[0]['lr'] = lr_init

    # monitor losses
    train_loss = 0
    valid_loss = 0
    
    num_train_correct_preds_per_domain = []
    num_valid_correct_preds_per_domain = []
    num_unseen_valid_correct_preds_per_domain = []
    tensor_list_losses_epoch_train = []
    tensor_list_losses_epoch_valid = []
    tensor_list_losses_epoch_valid_unseen = []
    
    # First term is ERM term during training, second is ERM term during validation and third is ERM term on unseen attacks
    # at validation
    loss_terms_train = np.array([0], dtype='f')
    loss_terms_valid = np.array([0], dtype='f')
    unseen_loss_terms_valid = np.array([0], dtype='f')
    if is_REx:
        # Adding a second entry to each, corresponding to the REx variance term added in the loss when using REx:
        # REx loss = ERM_term + REx_variance_term
        loss_terms_train = np.array([0, 0], dtype='f')
        loss_terms_valid = np.array([0, 0], dtype='f')
        unseen_loss_terms_valid = np.array([0, 0], dtype='f')
    
    which_batch_train = 1
    num_training_batches_in_epoch = len(train_loader)
    which_batch_valid = 1
    num_validation_batches = len(valid_loader)
         
    ###################
    # train the model #
    ###################
    model.train()
    for data,label in train_loader:
        data, label = data.to(device), label.to(device)
#         print(data.size())
        
        with ctx_noparamgrad_and_eval(model):
            # Generate MSD data
            list_data_all_domains = [data + msd_v0(device, model, data, label)]
            
            # Keeping the following in case we run MSD with something else
            num_domains = len(list_data_all_domains)
            # Initialise count of correct predictions and losses per domains
            if len(num_train_correct_preds_per_domain) == 0:
                num_train_correct_preds_per_domain = np.zeros(num_domains)
                tensor_list_losses_epoch_train = torch.zeros(num_domains).to(device)
            
            list_label_all_domains = [label] * num_domains
    
    
    
        optimizer.zero_grad()
        # calculate the loss
        loss = compute_loss(is_REx, beta, loss_terms_train, model, list_data_all_domains, list_label_all_domains,
                            num_domains, num_train_correct_preds_per_domain, tensor_list_losses_epoch_train)

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        
        # # Debugging
        # if which_batch_train % 10 == 0:
        #     print(time.time() - timer)
        #     print("Training, epoch ", epoch+1, ": done with batch ", which_batch_train, " out of ", num_training_batches_in_epoch, " loss = ", loss.item())
        #     timer = time.time()
        #     print("GPU memory allocated in GB:", torch.cuda.memory_allocated()/10**9)
        which_batch_train += 1
#         print(tensor_list_losses_epoch_train, '\n')
        


        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, label in valid_loader:
        data, label = data.to(device), label.to(device)
        
        with ctx_noparamgrad_and_eval(model):
            # Clean data is a domain
            list_data_all_domains = [data]
            
            # Generate domain corresponding to PGD Linf
            list_data_all_domains.append(adversary_PGD_Linf_std.perturb(data, label))
            
            # Generate domain corresponding to PGD L2
            list_data_all_domains.append(adversary_PGD_L2_std.perturb(data, label))
                        
            # Generate domain corresponding to PGD L1
            list_data_all_domains.append(adversary_PGD_L1_std.perturb(data, label))
            
            
            
            num_domains = len(list_data_all_domains)
            # Initialise count of correct predictions and losses per domains
            if len(num_valid_correct_preds_per_domain) == 0:
                num_valid_correct_preds_per_domain = np.zeros(num_domains)
                tensor_list_losses_epoch_valid = torch.zeros(num_domains).to(device)
            list_label_all_domains = [label] * num_domains
            
            
            # If loop to eval on unseen attacks at validation
            if use_unseen_attacks:
                # Generate data for unseen domains
                list_data_all_unseen_domains = []
                list_data_all_unseen_domains.append(adversary_PGD_Linf_unseen.perturb(data, label))
                list_data_all_unseen_domains.append(adversary_CW_unseen.perturb(data, label))
                list_data_all_unseen_domains.append(adversary_deepfool_unseen.perturb(data, label))
                list_data_all_unseen_domains.append(adversary_autoattack_unseen.run_standard_evaluation(data, label, bs=batch_size_train_and_valid))
                # Generate domain corresponding to CW that we skipped during training
                list_data_all_unseen_domains.append(adversary_CW.perturb(data, label))
                # Generate domain corresponding to Deepfool that we skipping during training
                list_data_all_unseen_domains.append(adversary_deepfool.perturb(data, label))

                
                num_unseen_domains = len(list_data_all_unseen_domains)
                # Initialise count of correct predictions and losses per domains
                if len(num_unseen_valid_correct_preds_per_domain) == 0:
                    num_unseen_valid_correct_preds_per_domain = np.zeros(num_unseen_domains)
                    tensor_list_losses_epoch_valid_unseen = torch.zeros(num_unseen_domains).to(device)
                list_label_all_unseen_domains = [label] * num_unseen_domains
                
                with torch.no_grad():
                    compute_loss(is_REx, beta, unseen_loss_terms_valid, model, list_data_all_unseen_domains, 
                                 list_label_all_unseen_domains, num_unseen_domains, num_unseen_valid_correct_preds_per_domain, 
                                 tensor_list_losses_epoch_valid_unseen)
                
                
        

        with torch.no_grad():
            loss = compute_loss(is_REx, beta, loss_terms_valid, model, list_data_all_domains, 
                                list_label_all_domains, num_domains, num_valid_correct_preds_per_domain, 
                                tensor_list_losses_epoch_valid)
        
        valid_loss += loss.item() * data.size(0)
        
        # Debugging
        if which_batch_valid % 10 == 0:
            print("Validation, epoch ", epoch+1, ": done with batch ", which_batch_valid, " out of ", num_validation_batches)
            print("GPU memory allocated in GB:", torch.cuda.memory_allocated()/10**9)
        which_batch_valid += 1
    
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.sampler)
    train_acc_per_domain = num_train_correct_preds_per_domain / len(train_loader.sampler)
    loss_terms_train = loss_terms_train / num_training_batches_in_epoch
    valid_loss = valid_loss / len(valid_loader.sampler)
    valid_acc_per_domain = num_valid_correct_preds_per_domain / len(valid_loader.sampler)
    loss_terms_valid = loss_terms_valid / num_validation_batches
    if use_unseen_attacks:
        unseen_loss_terms_valid = unseen_loss_terms_valid / num_validation_batches
    
    # calculate average losses per domain over epoch (so far the following losses were summed over all minibatches)
    tensor_list_losses_epoch_train = torch.div(tensor_list_losses_epoch_train, num_training_batches_in_epoch)
    list_losses_epoch_train = tensor_list_losses_epoch_train.tolist()
    tensor_list_losses_epoch_valid = torch.div(tensor_list_losses_epoch_valid, num_validation_batches)
    list_losses_epoch_valid = tensor_list_losses_epoch_valid.tolist()
    if use_unseen_attacks:
        tensor_list_losses_epoch_valid_unseen = torch.div(tensor_list_losses_epoch_valid_unseen, num_validation_batches)
        list_losses_epoch_valid_unseen = tensor_list_losses_epoch_valid_unseen.tolist()
    
    print('Epoch: {} \tTraining REx Loss: {:.6f} \tValidation REx Loss: {:.6f}'.format(
        epoch+1, 
        train_loss,
        valid_loss
        ))
    
    print("Epoch:", epoch+1, " \tTraining accuracy: ", train_acc_per_domain, " \tValidation accuracy: ", valid_acc_per_domain)

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). '.format(
        valid_loss_min,
        valid_loss))
        best_epoch = epoch
        
    if is_REx:
        path_of_checkpoint = TRAINED_MODEL_PATH + 'model_AIT_REx_' + str(epoch) + '.pt'
    else:
        path_of_checkpoint = TRAINED_MODEL_PATH + 'model_AIT_ERM_' + str(epoch) + '.pt'
    
    checkpoint = {'current_model': model.state_dict(),
                  'optimiser': optimizer.state_dict(),
                  'beta': beta,
                  'beta_prev': beta_prev,
                  'lr_init': lr_init,
                  'epoch': epoch + 1,
                  'best_epoch': best_epoch,
                  'seed': seed
                 }
    if epoch%5==0:
        torch.save(checkpoint, path_of_checkpoint)
    valid_loss_min = valid_loss
    
    writer.add_scalar('Learning_rate', lr_init, epoch+1)
    writer.add_scalar('Momentum', momentum, epoch+1)
    if is_REx:
        writer.add_scalar('beta', beta, epoch+1)
        writer.add_scalar('beta_prev', beta_prev, epoch+1)

    writer.add_scalar('Training_loss', train_loss, epoch+1)
    writer.add_scalar('Validation_loss', valid_loss, epoch+1)
    
    # TODO / WARNING: in the writer, for accuracies, the index of the attacks is hardcoded clean=0, PGD=1 etc.!
    writer.add_scalar('Training_accuracy_MSD', train_acc_per_domain[0], epoch+1)
    writer.add_scalar('Training_loss_MSD', list_losses_epoch_train[0], epoch+1)
    
    # writer.add_scalar('Training_accuracy_PGD_Linf', train_acc_per_domain[1], epoch+1)
    # writer.add_scalar('Training_loss_PGD_Linf', list_losses_epoch_train[1], epoch+1)
    
    # writer.add_scalar('Training_accuracy_PGD_L2', train_acc_per_domain[2], epoch+1)
    # writer.add_scalar('Training_loss_PGD_L2', list_losses_epoch_train[2], epoch+1)
    
    # writer.add_scalar('Training_accuracy_PGD_L1', train_acc_per_domain[3], epoch+1)
    # writer.add_scalar('Training_loss_PGD_L1', list_losses_epoch_train[3], epoch+1)
    
    writer.add_scalar('ERM_term_train', loss_terms_train[0], epoch+1)
    writer.add_scalar('ERM_term_validation', loss_terms_valid[0], epoch+1)
    if use_unseen_attacks:
        writer.add_scalar('unseen_ERM_term_validation', unseen_loss_terms_valid[0], epoch+1)
    if is_REx:
        writer.add_scalar('REx_variance_term_train', loss_terms_train[1], epoch+1)
        writer.add_scalar('REx_variance_term_validation', loss_terms_valid[1], epoch+1)
        if use_unseen_attacks:
            writer.add_scalar('unseen_REx_variance_term_validation', unseen_loss_terms_valid[1], epoch+1)
    
    
    
    writer.add_scalar('Validation_accuracy_clean', valid_acc_per_domain[0], epoch+1)
    writer.add_scalar('Validation_loss_clean', list_losses_epoch_valid[0], epoch+1)
    
    writer.add_scalar('Validation_accuracy_PGD_Linf', valid_acc_per_domain[1], epoch+1)
    writer.add_scalar('Validation_loss_PGD_Linf', list_losses_epoch_valid[1], epoch+1)
    
    writer.add_scalar('Validation_accuracy_PGD_L2', valid_acc_per_domain[2], epoch+1)
    writer.add_scalar('Validation_loss_PGD_L2', list_losses_epoch_valid[2], epoch+1)
    
    writer.add_scalar('Validation_accuracy_PGD_L1', valid_acc_per_domain[3], epoch+1)
    writer.add_scalar('Validation_loss_PGD_L1', list_losses_epoch_valid[3], epoch+1)
    
    
    
    
    if use_unseen_attacks:
        unseen_valid_acc_per_domain = num_unseen_valid_correct_preds_per_domain / len(valid_loader.sampler)
        writer.add_scalar('Validation_accuracy_PGD_Linf_unseen', unseen_valid_acc_per_domain[0], epoch+1)
        writer.add_scalar('Validation_loss_PGD_Linf_unseen', list_losses_epoch_valid_unseen[0], epoch+1)
        
        writer.add_scalar('Validation_accuracy_CW_unseen', unseen_valid_acc_per_domain[1], epoch+1)
        writer.add_scalar('Validation_loss_CW_unseen', list_losses_epoch_valid_unseen[1], epoch+1)
        
        writer.add_scalar('Validation_accuracy_Deepfool_unseen', unseen_valid_acc_per_domain[2], epoch+1)
        writer.add_scalar('Validation_loss_Deepfool_unseen', list_losses_epoch_valid_unseen[2], epoch+1)
        
        writer.add_scalar('Validation_accuracy_AutoAttack_unseen', unseen_valid_acc_per_domain[3], epoch+1)
        writer.add_scalar('Validation_loss_Autoattack_unseen', list_losses_epoch_valid_unseen[3], epoch+1)

        writer.add_scalar('Validation_accuracy_CW_base', unseen_valid_acc_per_domain[4], epoch+1)
        writer.add_scalar('Validation_loss_CW_base', list_losses_epoch_valid_unseen[4], epoch+1)
    
        writer.add_scalar('Validation_accuracy_Deepfool_base', unseen_valid_acc_per_domain[5], epoch+1)
        writer.add_scalar('Validation_loss_Deepfool_base', list_losses_epoch_valid_unseen[5], epoch+1)
    # END OF HARDCODED INDICES
    
    for name, params in model.named_parameters():
#         print(name, params.size())
        writer.add_histogram(f'grads/{name}', params.grad.data, epoch)
        writer.add_histogram(f'weights/{name}', params.data, epoch)
        
#     beta += 1

writer.close()