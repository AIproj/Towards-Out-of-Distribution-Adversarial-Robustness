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


# if str(device) == "cuda" and torch.cuda.device_count() > 1:
#     print("Using DataParallel")
#     model = torch.nn.DataParallel(model)
# model.to(device)













# divided by 10 eps, eps_iter and CW's lr, added as input binary_search_steps to CW attacks


adversary_PGD = attacks.LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
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

adversary_PGD_unseen = attacks.LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.4,
    nb_iter=40, eps_iter=0.033, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_deepfool_unseen = DeepfoolLinfAttack(
        model, num_classes=10, nb_iter=50, eps=0.4, clip_min=0.0, clip_max=1.0)

adversary_autoattack_unseen = AutoAttack(model, norm='Linf', eps=.3, 
        version='standard', seed=None, verbose=False)

adversary_PGD_L2_std = attacks.L2PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=2.,
    nb_iter=40, eps_iter=0.1, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)

adversary_PGD_L1_std = attacks.L1PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=10.,
    nb_iter=40, eps_iter=0.5, rand_init=True, clip_min=0.0,
    clip_max=1.0, targeted=False)



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





















    





waterfall = True
is_REx = True
resume = True
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
waterfall_epoch = 726
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr = lr_init, momentum=momentum)



if resume:
    checkpoint = torch.load("experiments/MNIST/MLP/pretrain_std/ERM_lr_init_0.01/model_AIT_ERM_725.pt")
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
        

TRAINED_MODEL_PATH = "experiments/MNIST/MLP/pretrain_std/"
if is_REx:
    TRAINED_MODEL_PATH += "REx"
else:
    TRAINED_MODEL_PATH += "ERM"
if waterfall:
    TRAINED_MODEL_PATH += "_waterfall"
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
            # Clean data is a domain
            list_data_all_domains = [data]
#             timer = time.time()
            
            # Generate domain corresponding to PGD
            list_data_all_domains.append(adversary_PGD.perturb(data, label))
            
            # Generate domain corresponding to CW
            list_data_all_domains.append(adversary_CW.perturb(data, label))
            
            # Generate domain corresponding to Deepfool
            list_data_all_domains.append(adversary_deepfool.perturb(data, label))
# #             print(time.time() - timer)

            
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
        
        # Debugging
        if which_batch_train % 10 == 0:
            print(time.time() - timer)
            print("Training, epoch ", epoch+1, ": done with batch ", which_batch_train, " out of ", num_training_batches_in_epoch, " loss = ", loss.item())
            timer = time.time()
            print("GPU memory allocated in GB:", torch.cuda.memory_allocated()/10**9)
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
            
            # Generate domain corresponding to PGD
            list_data_all_domains.append(adversary_PGD.perturb(data, label))
            
            # Generate domain corresponding to CW
            list_data_all_domains.append(adversary_CW.perturb(data, label))
                        
            # Generate domain corresponding to Deepfool
            list_data_all_domains.append(adversary_deepfool.perturb(data, label))
            
            
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
                list_data_all_unseen_domains.append(adversary_PGD_unseen.perturb(data, label))
                list_data_all_unseen_domains.append(adversary_CW_unseen.perturb(data, label))
                list_data_all_unseen_domains.append(adversary_deepfool_unseen.perturb(data, label))
                list_data_all_unseen_domains.append(adversary_autoattack_unseen.run_standard_evaluation(data, label, bs=batch_size_train_and_valid))
                # Generate domain corresponding to PGD L2
                list_data_all_unseen_domains.append(adversary_PGD_L2_std.perturb(data, label))  
                # Generate domain corresponding to PGD L1
                list_data_all_unseen_domains.append(adversary_PGD_L1_std.perturb(data, label))

                
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
    writer.add_scalar('Training_accuracy_clean', train_acc_per_domain[0], epoch+1)
    writer.add_scalar('Training_loss_clean', list_losses_epoch_train[0], epoch+1)
    
    writer.add_scalar('Training_accuracy_PGD', train_acc_per_domain[1], epoch+1)
    writer.add_scalar('Training_loss_PGD', list_losses_epoch_train[1], epoch+1)
    
    writer.add_scalar('Training_accuracy_CW', train_acc_per_domain[2], epoch+1)
    writer.add_scalar('Training_loss_CW', list_losses_epoch_train[2], epoch+1)
    
    writer.add_scalar('Training_accuracy_Deepfool', train_acc_per_domain[3], epoch+1)
    writer.add_scalar('Training_loss_Deepfool', list_losses_epoch_train[3], epoch+1)
    
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
    
    writer.add_scalar('Validation_accuracy_PGD', valid_acc_per_domain[1], epoch+1)
    writer.add_scalar('Validation_loss_PGD', list_losses_epoch_valid[1], epoch+1)
    
    writer.add_scalar('Validation_accuracy_CW', valid_acc_per_domain[2], epoch+1)
    writer.add_scalar('Validation_loss_CW', list_losses_epoch_valid[2], epoch+1)
    
    writer.add_scalar('Validation_accuracy_Deepfool', valid_acc_per_domain[3], epoch+1)
    writer.add_scalar('Validation_loss_Deepfool', list_losses_epoch_valid[3], epoch+1)
    
    
    
    if use_unseen_attacks:
        unseen_valid_acc_per_domain = num_unseen_valid_correct_preds_per_domain / len(valid_loader.sampler)
        writer.add_scalar('Validation_accuracy_PGD_unseen', unseen_valid_acc_per_domain[0], epoch+1)
        writer.add_scalar('Validation_loss_PGD_unseen', list_losses_epoch_valid_unseen[0], epoch+1)
        
        writer.add_scalar('Validation_accuracy_CW_unseen', unseen_valid_acc_per_domain[1], epoch+1)
        writer.add_scalar('Validation_loss_CW_unseen', list_losses_epoch_valid_unseen[1], epoch+1)
        
        writer.add_scalar('Validation_accuracy_Deepfool_unseen', unseen_valid_acc_per_domain[2], epoch+1)
        writer.add_scalar('Validation_loss_Deepfool_unseen', list_losses_epoch_valid_unseen[2], epoch+1)
        
        writer.add_scalar('Validation_accuracy_AutoAttack_unseen', unseen_valid_acc_per_domain[3], epoch+1)
        writer.add_scalar('Validation_loss_Autoattack_unseen', list_losses_epoch_valid_unseen[3], epoch+1)

        writer.add_scalar('Validation_accuracy_PGD_L2_unseen', unseen_valid_acc_per_domain[4], epoch+1)
        writer.add_scalar('Validation_loss_PGD_L2_unseen', list_losses_epoch_valid_unseen[4], epoch+1)

        writer.add_scalar('Validation_accuracy_PGD_L1_unseen', unseen_valid_acc_per_domain[5], epoch+1)
        writer.add_scalar('Validation_loss_PGD_L1_unseen', list_losses_epoch_valid_unseen[5], epoch+1)
    # END OF HARDCODED INDICES
    
    for name, params in model.named_parameters():
#         print(name, params.size())
        writer.add_histogram(f'grads/{name}', params.grad.data, epoch)
        writer.add_histogram(f'weights/{name}', params.data, epoch)
        
#     beta += 1

writer.close()