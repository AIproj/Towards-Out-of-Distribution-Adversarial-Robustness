import argparse
from distutils.util import strtobool
from math import pi

argument_parser = argparse.ArgumentParser()

argument_parser.add_argument("--lr_init", type=float, help="Optional. Initial learning rate value, default=0.1 if not resuming, last lr in checkpoint if resuming.")
argument_parser.add_argument("--beta", type=float, help="Optional. Overrides value of the coefficient of the REx term. Default: 5 for CIFAR10, or past value from checkpoint if resuming.")
argument_parser.add_argument("--force_scheduler_reset", type=lambda x: bool(strtobool(x)), help="Optional. Forces scheduler to reset, useful when starting REx for the first time. \
    Recommended to use a custom --lr_init when using this. Default: False.")
argument_parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), help="Optional. Please set to True to resume. Consider setting a --resume_path. Default: False.")
argument_parser.add_argument("--wd", type=float, help="Optional. Weight decay value for the optimiser. Default: 5e-4 for CIFAR10, or past value from checkpoint if resuming.")
argument_parser.add_argument("--output_suffix", type=str, help="Optional. Suffix for path where files should be created.")
# Careful: the arg below doesn't affect where all files created during training will be saved.
argument_parser.add_argument("--resume_path", type=str, help="Optional. Enter full path to the checkpoint as aaa/bbb/ccc. Default if not provided: \
    load automatically the last checkpoint given the --seen_domains and --REx. This does not affect where the files will be written during training !")
required_args = argument_parser.add_argument_group("required arguments")
required_args.add_argument("--seen_domains", required=True, type=str, help="Please enter the seen domains as either MSD, \n PGDs for PGD L1, PGD L2 and PGD Linf, \n\
            var for PGD Linf, Deepfool and Carlini&Wagner, \n PGD_Linf, \n PGD_L2, \n PGD_L1, \n clean. ")
required_args.add_argument("--REx", required=True, type=lambda x: bool(strtobool(x)), help="Please pass --REx=True for REx.")

parsed_args = argument_parser.parse_args()

from ast import Raise
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

import os, random




# Make sure validation splits are the same at all time (e.g. even after loading)
seed = 0

# See https://stackoverflow.com/questions/60311307/how-does-one-reset-the-dataloader-in-pytorch for a discussion on how to keep
# the same data when using a dataloader (here, we use this for validation and testing for which we enumerate the dataloader to
# ensure that we access the same data across multiple runs/epochs since we don't use the full data to speed up computations)
def seed_init_fn(seed=seed):
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   return

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_workers = 0
# Make sure test_data is a multiple of batch_size_test
batch_size_train_and_valid = 128
batch_size_test = 200

# proportion of full training set used for validation
valid_size = 0.2




transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_to_tensor = transforms.ToTensor()

train_and_valid_data = datasets.CIFAR10(root = 'data', train = True, download = True, transform = transform_train)
test_data = datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_to_tensor)




# transform = transforms.ToTensor()
# train_and_valid_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
# test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)

num_valid_samples = int(np.floor(valid_size * len(train_and_valid_data)))
num_train_samples = len(train_and_valid_data) - num_valid_samples
train_data, valid_data = torch.utils.data.random_split(train_and_valid_data, [num_train_samples, num_valid_samples], generator=torch.Generator().manual_seed(seed))

train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size_train_and_valid, worker_init_fn=seed_init_fn)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size_train_and_valid, worker_init_fn=seed_init_fn)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size_test, worker_init_fn=seed_init_fn)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
#         print(x.size(), out.size())
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])









model = ResNet18()
model.to(device)


# Note: the base, pre-trained model was trained in 60 epochs with no weight decay or schedule.
model.load_state_dict(torch.load('model_ResNet18.pt'))
model.to(device)










# divided by 10 eps, eps_iter and CW's lr, added as input binary_search_steps to CW attacks


adversary_PGD_Linf_std = attacks.LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
    nb_iter=10, eps_iter=2/255, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_PGD_L2_std = attacks.L2PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.5,
    nb_iter=10, eps_iter=15/255, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_PGD_L1_std = attacks.L1PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=10.,
    nb_iter=10, eps_iter=20/255, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_CW = attacks.CarliniWagnerL2Attack(
    model, num_classes=10, max_iterations=20, learning_rate=0.01,
    binary_search_steps=5, clip_min=0.0, clip_max=1.0)

adversary_deepfool = DeepfoolLinfAttack(
        model, num_classes=10, nb_iter=30, eps=0.011, clip_min=0.0, clip_max=1.0)

# Unseen attacks used for validation, has bigger learning rate and number of iterations. CHANGED PGD Linf eps iter to 12/255 AND CW LR to 0.0115
adversary_CW_unseen = attacks.CarliniWagnerL2Attack(
    model, num_classes=10, max_iterations=30, learning_rate=0.012,
    binary_search_steps=7, clip_min=0.0, clip_max=1.0)

adversary_PGD_Linf_unseen = attacks.LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=12/255,
    nb_iter=10, eps_iter=2/255, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_deepfool_unseen = DeepfoolLinfAttack(
        model, num_classes=10, nb_iter=50, eps=8/255, clip_min=0.0, clip_max=1.0)

adversary_autoattack_unseen = AutoAttack(model, norm='Linf', eps=8/255, 
        version='standard', seed=None, verbose=False)







### The following functions (up to msd_v0) are taken from the repo of "Adversarial Robustness Against the Union of Multiple Perturbation Models"
### See https://github.com/locuslab/robust_union/
def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

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
    y = y.view(-1,3,32,32)
    y *= x.sign()
    return y

def msd_v0(device, model, X,y, epsilon_l_inf = 8/255, epsilon_l_2= 0.5, epsilon_l_1 = 10, 
                alpha_l_inf = 2/255, alpha_l_2 = 15/255, alpha_l_1 = 20/255, num_iter = 10):
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





def generate_domains(domain_name, data, label, batch_size=batch_size_test, bool_correct_preds_per_domain={}):
    if len(bool_correct_preds_per_domain) == 0:
        mask = torch.ones_like(label)
    else:
        mask = bool_correct_preds_per_domain[domain_name]
    masked_data = data[mask, :, :, :]
    masked_label = label[mask]

    # All the data might have been masked. In that case return None.
    if len(masked_data) == 0:
        return None

    if domain_name == 'clean':
        return masked_data
    if domain_name == 'PGD_L1_std':
        return adversary_PGD_L1_std.perturb(masked_data, masked_label)
    if domain_name == 'PGD_L2_std':
        return adversary_PGD_L2_std.perturb(masked_data, masked_label)
    if domain_name == 'PGD_Linf_std':
        return adversary_PGD_Linf_std.perturb(masked_data, masked_label)
    if domain_name == 'Deepfool_base':
        return adversary_deepfool.perturb(masked_data, masked_label)
    if domain_name == "CW_base":
        return adversary_CW.perturb(masked_data, masked_label)
    if domain_name == 'PGD_Linf_mod':
        return adversary_PGD_Linf_unseen.perturb(masked_data, masked_label)
    if domain_name == 'Deepfool_mod':
        return adversary_deepfool_unseen.perturb(masked_data, masked_label)
    if domain_name == 'CW_mod':
        return adversary_CW_unseen.perturb(masked_data, masked_label)
    if domain_name == "Autoattack":
        return adversary_autoattack_unseen.run_standard_evaluation(masked_data, masked_label, bs=len(masked_label))
    if domain_name == "MSD":
        return masked_data + msd_v0(device, model, masked_data, masked_label)








# Further possible improvement: avoid making loss_helper() and compute_loss() update the mutable loss terms (ERM and REx terms)
# and losses_over_epoch (the accumulated losses over each domain during the epoch) in the method. Not "pythonic".
def loss_helper(model, data_all_domains, label, domains, losses_over_epoch):
    # Could use a dic instead of list if we ever need to keep track of the individual losses for further processing.
    temp_losses = []
    for domain in domains:
        preds = model(data_all_domains[domain])
        individual_losses_temp = F.cross_entropy(preds, label)
        temp_losses.append(individual_losses_temp)
        losses_over_epoch[domain] += individual_losses_temp
    
    tensor_losses = torch.stack(temp_losses)
    ERM_term = torch.sum(tensor_losses) / len(domains)
    REx_variance_term = torch.var(tensor_losses)
    return ERM_term, REx_variance_term

def compute_loss(is_REx, beta, loss_terms, model, data_all_domains, label, domains, losses_over_epoch):
    if is_REx:
        ERM_term, REx_variance_term = loss_helper(model, data_all_domains, label, domains, losses_over_epoch)
        loss_terms_temp = [ERM_term.item(), REx_variance_term.item()]
        loss_terms += np.array(loss_terms_temp)
        loss = REx_loss(ERM_term, REx_variance_term, beta)
    else:
        ERM_term, _ = loss_helper(model, data_all_domains, label, domains, losses_over_epoch)
        loss_terms += np.array([ERM_term.item()])
        loss = ERM_term
    return loss


def REx_loss(ERM_term, REx_variance_term, beta):
    return beta * REx_variance_term + ERM_term

 


# Keep track across restarts of which samples were still correctly predicted, for each attack
def track_correct_pred_per_domain(model, data_all_domains, labels, domains, bool_correct_per_domain):
    for domain in domains:
        # Case when the mask filtered all data
        if data_all_domains[domain] == None:
            continue

        preds = model(data_all_domains[domain])
        # bool_correct_per_domain[domain] = torch.logical_and(bool_correct_per_domain[domain], (torch.argmax(preds, dim=1) == label_all_domains[domain]))

        # Array sizes of preds and bool_correct are different because of the mask when generating the domains, so handling it manually. Maybe
        # there is/will be a native method to handle this but gotta go fast.
        mask = bool_correct_per_domain[domain]
        are_preds_right = (torch.argmax(preds, dim=1) == labels[mask])
        i = 0
        for k in range(len(bool_correct_per_domain[domain])):
            if bool_correct_per_domain[domain][k]:
                bool_correct_per_domain[domain][k] = are_preds_right[i]
                i += 1
    return

# Compute the number of correct predictions against each attack after all the restarts
def update_num_correct_pred_per_domain(num_correct_per_domain, bool_correct_per_domain, domains):
    for domain in domains:
        num_correct_per_domain[domain] += bool_correct_per_domain[domain].sum().item()
    return

# Compute the number of correct predictions if the attacker was using an ensemble of all attacks. Skip the attacks in skipped_domains_worst_case from calculation.
def get_num_correct_worst_case(bool_correct_per_domain, domains, skipped_domains_worst_case=[]):
    # TODO WARNING
    # TODO WARNING
    if len(domains) == 0:
        raise ValueError("No domain has been defined !")
    
    bool_correct_worst_case = torch.ones_like(bool_correct_per_domain[domains[0]], dtype=torch.bool)
    for domain in domains:
        if domain in skipped_domains_worst_case:
            continue
        bool_correct_worst_case = torch.logical_and(bool_correct_worst_case, bool_correct_per_domain[domain])

    return bool_correct_worst_case.sum().item()

# Get which attacks were seen based on model filename. 
def get_seen_domains_from_filename(model_name):
    split_model_name = model_name.split('_')
    seen_attacks = []
    if "MSD" in split_model_name:
        if "ERM" in split_model_name:
            seen_attacks = ['PGD_L1_std', 'PGD_L2_std', 'PGD_Linf_std']
        else:
            seen_attacks = ['clean', 'PGD_L1_std', 'PGD_L2_std', 'PGD_Linf_std']
    if "PGDs" in split_model_name:
        seen_attacks = ['clean', 'PGD_L1_std', 'PGD_L2_std', 'PGD_Linf_std']
    if "std" in split_model_name:
        seen_attacks = ['clean', 'PGD_Linf_std', 'Deepfool_base', 'CW_base']
    if "clean" in split_model_name:
        seen_attacks = ['clean']
    if "L1" in split_model_name:
        seen_attacks = ['clean', 'PGD_L1_std']
    if "L2" in split_model_name:
        seen_attacks = ['clean', 'PGD_L2_std']
    if "Linf" in split_model_name:
        seen_attacks = ['clean', 'PGD_Linf_std']
    return seen_attacks


# Find the checkpoint with highest number of epochs in a given folder
def find_last_checkpoint(path):
    for root, dirs, files in os.walk(path):
        last_checkpoint = max([(len(file), file) for file in files if file.endswith(".pt")])[1]
        return last_checkpoint

# Generate list of seen domains based on parsed --REx and --seen_domains.
def get_seen_domains(user_input_seen_domains, is_REx):
    if user_input_seen_domains == "MSD":
        if is_REx:
            seen_domains = ['clean', 'PGD_L1_std', 'PGD_L2_std', 'PGD_Linf_std']
        else:
            seen_domains = ['MSD']
    elif user_input_seen_domains == "PGDs":
        seen_domains = ['clean', 'PGD_L1_std', 'PGD_L2_std', 'PGD_Linf_std']
    elif user_input_seen_domains == "var":
        seen_domains = ['clean', 'PGD_Linf_std', 'Deepfool_base', 'CW_base']
    elif user_input_seen_domains == "PGD_L1":
        seen_domains = ['PGD_L1_std']
    elif user_input_seen_domains == "PGD_L2":
        seen_domains = ['PGD_L2_std']
    elif user_input_seen_domains == "PGD_Linf":
        seen_domains = ['PGD_Linf_std']
    elif user_input_seen_domains == "clean":
        seen_domains = ['clean']
    else:
        raise ValueError("Error. Please enter the seen domains as either MSD, \n PGDs for PGD L1, PGD L2 and PGD Linf, \n\
            var for PGD Linf, Deepfool and Carlini&Wagner, \n PGD_Linf, \n PGD_L2, \n PGD_L1. \n\
                You entered: ", user_input_seen_domains)
    return seen_domains

def xstr(s):
    if s is None:
        return ''
    else:
        return s

# Careful: this is a cos schedule that starts at the min and progresses to the max after half a period due to - sign on cos
def inv_cos_schedule(period, minimum, maximum, epoch):
    return minimum + 0.5 * (maximum - minimum) * (1 - np.cos(2 * np.pi * epoch / period))









waterfall = False
is_REx = parsed_args.REx
resume = False
use_unseen_attacks = False
save_interval = 1
# Controls how often we also validate on unseen domains. Note: this should never be used for model selection otherwise unseen attacks are not unseen !
full_valid_interval = 5

all_domains = ['clean', 'PGD_L1_std', 'PGD_L2_std', 'PGD_Linf_std', 'Deepfool_base', 'CW_base',
                'PGD_Linf_mod', 'Deepfool_mod', 'CW_mod', 'Autoattack']

seen_domains_training = get_seen_domains(parsed_args.seen_domains, is_REx)
# Handle the fact that MSD does technically see the Lp PGD, for validation, so we can compare it to e.g. the PGDs modality of training.
# This allows computing metrics on PGDs attacks as "seen" attacks for validation for MSD.
if parsed_args.seen_domains == "MSD" and not is_REx:
    seen_domains_validation = ['PGD_L1_std', 'PGD_L2_std', 'PGD_Linf_std']
else:
    seen_domains_validation = seen_domains_training
unseen_domains = [domain for domain in all_domains if domain not in seen_domains_validation]

# number of epochs to train the model
n_epochs_AIT = 1001
# initialize tracker for minimum validation loss
valid_loss_min = np.Inf  # set initial "min" to infinity
beta = 10
beta_prev = beta
lr_init = 0.1 # / beta
best_epoch = 0
starting_epoch = 0
# # the following var decides when we start waterfalling in the REx term (includes the chosen epoch)
# waterfall_epoch = 326
momentum = 0.9
if parsed_args.wd is not None:
    weight_decay = parsed_args.wd
else:
    weight_decay = 5e-4
optimizer = torch.optim.SGD(model.parameters(), lr = lr_init, momentum=momentum, weight_decay=weight_decay)
schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
# Controls the number of restart. Not necessary in general during training. 1 = attacks only computed once.
num_attack_restarts = 1
# Limit number of validation batches to the value below to save compute. Set to 0 to use the whole validation set.
limited_max_num_validation_batches = 10



        
# try:
#     if parsed_args.REx:
#         is_REx = True
# except:
#     raise ValueError("No --REx argument passed.")


if parsed_args.beta is not None:
    beta = parsed_args.beta
else:
    print("No --beta argument passed.")


TRAINED_MODEL_PATH = "experiments/CIFAR10/ResNet18/train/wd/" + xstr(parsed_args.output_suffix) + parsed_args.seen_domains
if is_REx:
    TRAINED_MODEL_PATH += "_REx_beta" + str(beta) + "/"
else:
    TRAINED_MODEL_PATH += "_ERM/"


if parsed_args.resume:
    resume = parsed_args.resume
    if parsed_args.resume_path is not None:
        resume_path = parsed_args.resume_path
    else:
        print('No arg was provided for the checkpoint to resume from. Resuming based on --seen_domains and --REx.')
        resume_path = TRAINED_MODEL_PATH + find_last_checkpoint(TRAINED_MODEL_PATH)
    try:
        checkpoint = torch.load(resume_path)
    except:
        raise ValueError("Cannot load from checkpoint at path " + str(resume_path) + "\n Consider using a --resume_path arg if not already doing so.")
    starting_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['current_model'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimiser'])
    schedule.load_state_dict(checkpoint['schedule'])
    lr_init = checkpoint['learning_rate']
    if parsed_args.beta is None:
        beta = checkpoint['beta']
        beta_prev = checkpoint['beta_prev']
        print("Beta and beta_prev loaded from default of resp ", beta, " and ", beta_prev) 
    else:
        print("Overriding beta value from checkpoint with value provided with --beta.")
    best_epoch = checkpoint['best_epoch']
    # is_REx = checkpoint['is_REx']
else:
    print("The user did not set --resume to True. Training from scratch...")


# MODIFY GIVEN PARSED ARGUMENTS
try:
    if parsed_args.lr_init is not None:
        lr_init = parsed_args.lr_init
        optimizer.param_groups[0]['lr'] = lr_init
        print("lr_init set to %f by user" % parsed_args.lr_init)
        if resume:
            print("WARNING: an argument was passed to override the lr_init loaded from the checkpoint !")
        
except:
    print("No learning rate passed as argument; using default/past value of ", lr_init)




# Reset LR to 0.1 and scheduler state for REx forced schedule
try:
    if parsed_args.force_scheduler_reset:
        # Careful, this resets accumulated momentum.
        optimizer = torch.optim.SGD(model.parameters(), lr = lr_init, momentum=momentum, weight_decay=weight_decay)
        # Used to be milestones = [4] for resume at 50 with MSD
        schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        print("Forced an opt/scheduler reset.")
    else:
        print("Did not force a scheduler reset.")
except:
    print("Did not force a scheduler reset.")

path_of_checkpoint = ""
writer = SummaryWriter(TRAINED_MODEL_PATH)

# Parallelise model if possible
if str(device) == "cuda" and torch.cuda.device_count() > 1:
    print("Using DataParallel")
    model = torch.nn.DataParallel(model)

for epoch in range(starting_epoch, n_epochs_AIT):
    # Only consider unseen attacks every full_valid_interval epochs
    if (epoch+1) % full_valid_interval == 0: #or epoch+1 >= 95:
        use_unseen_attacks = True
    else:
        use_unseen_attacks = False

    # Uncomment to use a cos schedule to increase beta. Very quick experiments on our end did not find much use for it.
    # beta = inv_cos_schedule(period=200, minimum=3, maximum=17, epoch=(epoch-starting_epoch))

    # Change domains used in validation based on whether unseen attacks are used
    if use_unseen_attacks:
        valid_domains = all_domains
    else:
        valid_domains = seen_domains_validation

    if is_REx:
        # First var is ERM term during training, second is ERM term during validation and third is ERM term on unseen attacks
        # at validation.
        # If is_REx, adding a second entry to each, corresponding to the REx variance term added in the loss when using REx:
        # REx loss = ERM_term + REx_variance_term
        loss_terms_train = np.array([0, 0], dtype='f')
        loss_terms_valid = np.array([0, 0], dtype='f')
        # unseen_loss_terms_valid = np.array([0, 0], dtype='f')
    else:
        loss_terms_train = np.array([0], dtype='f')
        loss_terms_valid = np.array([0], dtype='f')
        # unseen_loss_terms_valid = np.array([0], dtype='f')


    # Number of correct predictions on each domain
    num_train_correct_preds_per_domain = {}
    num_valid_correct_preds_per_domain = {}
    # Keep track of average ERM loss *per domain*
    training_losses_over_epoch = {}
    valid_losses_over_epoch = {}

    for domain in seen_domains_training:
        # results[domain] = 0
        # results[domain + "_bool_track_correct_preds"] = []
        training_losses_over_epoch[domain] = 0
        num_train_correct_preds_per_domain[domain] = 0
    
    for domain in valid_domains:
        valid_losses_over_epoch[domain] = 0
        num_valid_correct_preds_per_domain[domain] = 0
    num_valid_correct_preds_per_domain['worst_seen'] = 0

    # Keep track of total (REx or ERM) losses
    train_loss = 0
    valid_loss = 0

    which_batch_train = 1
    num_training_batches_in_epoch = len(train_loader)
    which_batch_valid = 1
    num_validation_batches = len(valid_loader)
    # Turn off limited_max_num_validation_batches (by setting it to 0) if == num_validation_batches
    if num_validation_batches == limited_max_num_validation_batches:
        limited_max_num_validation_batches = 0

    ###################
    # Train the model #
    ###################
    model.train()
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
#         print(data.size())
        
        # Keeps track for each sample and each domain of whether one restart succeeded in fooling the network by using logical and
        # on (label == prediction) and bool_track_correct_pred each iteration. 
        bool_track_correct_pred_per_domain = {}
        for domain in seen_domains_training:
            bool_track_correct_pred_per_domain[domain] = torch.ones_like(label, dtype=torch.bool)

        data_all_domains = {}
        for i_restarts in range(0, num_attack_restarts):
            with ctx_noparamgrad_and_eval(model):
                # Clean data is a domain.
                data_all_domains_current_restart = {}
                for domain in seen_domains_training:
                    data_all_domains_current_restart[domain] = generate_domains(domain, data, label, batch_size=batch_size_train_and_valid, bool_correct_preds_per_domain=bool_track_correct_pred_per_domain)
            with torch.no_grad():
                # Update data_all_domains's unsuccessful adversarial examples with new restart's ones. Could be improved very slightly by only updating with successful ones from the new restart.
                # Need to handle first iteration differently as a.masked_scatter would try to operate on an undefined var.
                for domain in seen_domains_training:
                    if i_restarts == 0:
                        data_all_domains[domain] = data_all_domains_current_restart[domain]
                    else:
                        data_all_domains[domain].masked_scatter(bool_track_correct_pred_per_domain[domain], data_all_domains_current_restart[domain])
                track_correct_pred_per_domain(model, data_all_domains_current_restart, label, seen_domains_training, bool_track_correct_pred_per_domain)


        optimizer.zero_grad()
        # calculate the loss
        loss = compute_loss(is_REx=is_REx, beta=beta, loss_terms=loss_terms_train, model=model, data_all_domains=data_all_domains, label=label,
                                domains=seen_domains_training, losses_over_epoch=training_losses_over_epoch)

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        
        # Update count of number of correct predictions per domain
        with torch.no_grad():
            update_num_correct_pred_per_domain(num_train_correct_preds_per_domain, bool_track_correct_pred_per_domain, seen_domains_training)

        if which_batch_train % 100 == 0:
            # print(time.time() - timer)
            print("Training, epoch ", epoch+1, ": done with batch ", which_batch_train, " out of ", num_training_batches_in_epoch, " loss = ", loss.item())
            # timer = time.time()
            # Track potential memory leaks
            # print("GPU memory allocated in GB:", torch.cuda.memory_allocated()/10**9)
        which_batch_train += 1
#         print(tensor_list_losses_epoch_train, '\n')



    ######################    
    # Validate the model #
    ######################
    model.eval()
    for _, (data, label) in enumerate(valid_loader):
        data, label = data.to(device), label.to(device)
#         print(data.size())
        
        # Keeps track for each sample and each domain of whether one restart succeeded in fooling the network by using logical and
        # on (label == prediction) and bool_track_correct_pred each iteration. 
        bool_track_correct_pred_per_domain = {}
        for domain in valid_domains:
            bool_track_correct_pred_per_domain[domain] = torch.ones_like(label, dtype=torch.bool)

        data_all_domains = {}
        for i_restarts in range(0, num_attack_restarts):
            with ctx_noparamgrad_and_eval(model):
                # Clean data is a domain.
                data_all_domains_current_restart = {}
                for domain in valid_domains:
                    data_all_domains_current_restart[domain] = generate_domains(domain, data, label, batch_size=batch_size_train_and_valid, bool_correct_preds_per_domain=bool_track_correct_pred_per_domain)
            with torch.no_grad():
                # Update data_all_domains's unsuccessful adversarial examples with new restart's ones. Could be improved very slightly by only updating with successful ones from the new restart.
                # Need to handle first iteration differently as a.masked_scatter would try to operate on an undefined var.
                for domain in valid_domains:
                    if i_restarts == 0:
                        data_all_domains[domain] = data_all_domains_current_restart[domain]
                    else:
                        data_all_domains[domain].masked_scatter(bool_track_correct_pred_per_domain[domain], data_all_domains_current_restart[domain])
                track_correct_pred_per_domain(model, data_all_domains_current_restart, label, valid_domains, bool_track_correct_pred_per_domain)


        # Calculate loss
        with torch.no_grad():
            loss = compute_loss(is_REx=is_REx, beta=beta, loss_terms=loss_terms_valid, model=model, data_all_domains=data_all_domains, label=label,
                                    domains=valid_domains, losses_over_epoch=valid_losses_over_epoch)
            update_num_correct_pred_per_domain(num_valid_correct_preds_per_domain, bool_track_correct_pred_per_domain, valid_domains)
            num_valid_correct_preds_per_domain['worst_seen'] += get_num_correct_worst_case(bool_correct_per_domain=bool_track_correct_pred_per_domain, domains=seen_domains_validation)

        valid_loss += loss.item() * data.size(0)



        # WARNING
        # WARNING
        # WARNING
        # We only use the first limited_max_num_validation_batches validation minibatches to save time given how long computing a dozen attacks is. This is
        # why ensuring consistency of the dataloader sampling was important, so we always see the same data every epoch.
        print("Validating, epoch ", epoch+1, ": done with batch ", which_batch_valid, " out of ", min(x for x in [num_validation_batches, limited_max_num_validation_batches] if x != 0))
        if which_batch_valid % 10 == 0:
            # print("Testing, epoch ", starting_epoch, ": done with batch ", which_batch_test, " out of ", num_test_batches)
            print("GPU memory allocated in GB:", torch.cuda.memory_allocated()/10**9, flush=True)
        # Only compute on the first limited_max_num_validation_batches minibatches
        if which_batch_valid == limited_max_num_validation_batches:
            break
        which_batch_valid += 1


    # Average loss over epoch. Careful about computations where we average over batches; they will have a bias if dataset size not multiple of batch_size
    train_loss = train_loss / len(train_loader.sampler)
    loss_terms_train = loss_terms_train / num_training_batches_in_epoch
    # Handling validation terms differently based on stopping early or not (because when using the full set its size may not be divisible by batch_size) !
    if which_batch_valid == limited_max_num_validation_batches:
        num_valid_samples_used = which_batch_valid * batch_size_train_and_valid
    else: 
        num_valid_samples_used = len(valid_loader.sampler)
    valid_loss = valid_loss / num_valid_samples_used
    loss_terms_valid = loss_terms_valid / which_batch_valid
    training_acc_per_domain = {}
    valid_acc_per_domain = {}
    for domain in seen_domains_training:
        training_acc_per_domain[domain] = num_train_correct_preds_per_domain[domain] / len(train_loader.sampler)
        training_losses_over_epoch[domain] = training_losses_over_epoch[domain] / num_training_batches_in_epoch
    for domain in valid_domains:
        valid_acc_per_domain[domain] = num_valid_correct_preds_per_domain[domain] / num_valid_samples_used
        valid_losses_over_epoch[domain] = valid_losses_over_epoch[domain] / which_batch_valid
    valid_seen_worst_case_acc = num_valid_correct_preds_per_domain['worst_seen'] / num_valid_samples_used

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, 
        train_loss,
        valid_loss
        ))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). '.format(
        valid_loss_min,
        valid_loss))
        best_epoch = epoch
        valid_loss_min = valid_loss
        
    if is_REx:
        path_of_checkpoint = TRAINED_MODEL_PATH + 'model_REx_' + str(epoch+1) + '.pt'
    else:
        path_of_checkpoint = TRAINED_MODEL_PATH + 'model_ERM_' + str(epoch+1) + '.pt'
    
    epoch_lr = schedule.get_last_lr()[0]
    schedule.step()
    # lr to start from in checkpoint
    lr_init = schedule.get_last_lr()[0]
    
    checkpoint = {'current_model': model.module.state_dict(),
                  'optimiser': optimizer.state_dict(),
                  'schedule': schedule.state_dict(),
                  'beta': beta,
                  'beta_prev': beta_prev,
                  'learning_rate': lr_init,
                  'epoch': epoch + 1,
                  'best_epoch': best_epoch,
                  'seed': seed
                 }

    if (epoch+1) % save_interval == 0: #and epoch+1 >= 50:
        torch.save(checkpoint, path_of_checkpoint)
    
    writer.add_scalar('Learning_rate', epoch_lr, epoch+1)
    writer.add_scalar('Momentum', momentum, epoch+1)
    writer.add_scalar('Weight_decay', weight_decay, epoch+1)

    if is_REx:
        writer.add_scalar('beta', beta, epoch+1)
        writer.add_scalar('beta_prev', beta_prev, epoch+1)

    writer.add_scalar('Training_loss', train_loss, epoch+1)
    writer.add_scalar('Validation_loss', valid_loss, epoch+1)

    writer.add_scalar('ERM_term_train', loss_terms_train[0], epoch+1)
    writer.add_scalar('ERM_term_validation', loss_terms_valid[0], epoch+1)

    if is_REx:
        writer.add_scalar('REx_variance_term_train', loss_terms_train[1], epoch+1)
        writer.add_scalar('REx_variance_term_validation', loss_terms_valid[1], epoch+1)

    valid_seen_acc = 0
    valid_seen_loss = 0
    valid_seen_list_losses = []
    for domain in seen_domains_validation:
        valid_seen_acc += valid_acc_per_domain[domain]
        valid_seen_loss += valid_losses_over_epoch[domain]
        valid_seen_list_losses.append(valid_losses_over_epoch[domain].detach().cpu())
    valid_seen_acc = valid_seen_acc / len(seen_domains_validation)
    valid_seen_loss = valid_seen_loss / len(seen_domains_validation)
    writer.add_scalar('Validation_seen_accuracy', valid_seen_acc, epoch+1)
    writer.add_scalar('Validation_seen_ERM_loss', valid_seen_loss, epoch+1)
    writer.add_scalar('Validation_seen_variance_term', np.var(valid_seen_list_losses), epoch+1)
    writer.add_scalar('Validation_seen_worst_case_accuracy', valid_seen_worst_case_acc, epoch+1)
    if is_REx:
        writer.add_scalar('Validation_seen_REx_loss', valid_seen_loss + beta*np.var(valid_seen_list_losses), epoch+1)
    if use_unseen_attacks:
        valid_unseen_acc = 0
        valid_unseen_loss = 0
        valid_unseen_list_losses = []
        for domain in unseen_domains:
            valid_unseen_acc += valid_acc_per_domain[domain]
            valid_unseen_loss += valid_losses_over_epoch[domain]
            valid_unseen_list_losses.append(valid_losses_over_epoch[domain].detach().cpu())
        valid_unseen_acc = valid_unseen_acc / len(unseen_domains)
        valid_unseen_loss = valid_unseen_loss / len(unseen_domains)
        writer.add_scalar('Validation_unseen_accuracy', valid_unseen_acc, epoch+1)
        writer.add_scalar('Validation_unseen_ERM_loss', valid_unseen_loss, epoch+1)
        writer.add_scalar('Validation_unseen_variance_term', np.var(valid_unseen_list_losses), epoch+1)
        if is_REx:
            writer.add_scalar('Validation_unseen_REx_loss', valid_unseen_loss + beta*np.var(valid_unseen_list_losses), epoch+1)


    for domain in seen_domains_training:
        writer.add_scalar('Training_accuracy_'+domain, training_acc_per_domain[domain], epoch+1)
        writer.add_scalar('Training_loss_'+domain, training_losses_over_epoch[domain], epoch+1)
    
    for domain in valid_domains:
        writer.add_scalar('Validation_accuracy_'+domain, valid_acc_per_domain[domain], epoch+1)
        writer.add_scalar('Validation_loss_'+domain, valid_losses_over_epoch[domain], epoch+1)

writer.close()