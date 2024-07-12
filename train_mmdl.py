#adapted from MMDL/ trainc.py, testc.py and mainc.py by DWB

import argparse

#model file
from Netc import Cnn_With_Clinical_Net

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import functional as F

import time
import copy
import os
from pathlib import Path  

from utils.custom_dset import CustomDset

from torchvision import datasets, models, transforms

import pandas as pd
import numpy as np
from sklearn import preprocessing

from torch.func import functional_call, vmap, grad


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def process_probs (model_output, names, labels, p_dict,pt_wts):
    probability = F.softmax(model_output, dim=1).data.squeeze()    
    probs = probability.cpu().numpy()
    for i in range(labels.size(0)):
        p = names[i]
        if p not in p_dict.keys():
            p_dict[p] = {
                'prob_0': 0, 
                'prob_1': 0,
                'label': labels[i].item(),      
                'img_num': 0,
                'weight':pt_wts[p]}
        if probs.ndim == 2:
            p_dict[p]['prob_0'] += probs[i, 0]
            p_dict[p]['prob_1'] += probs[i, 1]
            p_dict[p]['img_num'] += 1
        else:
            p_dict[p]['prob_0'] += probs[0]
            p_dict[p]['prob_1'] += probs[1]
            p_dict[p]['img_num'] += 1

def process_loss (model_loss, names,p_loss,pt_wts):
    loss = model_loss.detach().cpu().numpy()
    for i in range(loss.size):
        l = names[i]
        if l not in p_loss.keys(): 
          p_loss[l] = {
            "img_loss" : pt_wts[l]*loss[i],
            "img_num" : 0}
        if loss.ndim == 2:
            p_loss[l]["img_loss"] += pt_wts[l]*loss[i,0]
            p_loss[l]["img_num"] += 1
        else:
            p_loss[l]["img_loss"] += pt_wts[l]*loss[i]
            p_loss[l]["img_num"] += 1            

def tabulate_probs (p_dict):
    y_true = []
    y_pred = []
    score_list = []
    preid_list = []

    total = len(p_dict)
    correct = 0
    for key in p_dict.keys():
        preid_list.append(key)
        predict = 0
        if p_dict[key]['prob_0'] < p_dict[key]['prob_1']:
            predict = 1
        if p_dict[key]['label'] == predict:
            correct += 1
        y_true.append(p_dict[key]['label'])
        score_list.append([(p_dict[key]['prob_0']/p_dict[key]["img_num"]),(p_dict[key]['prob_1']/p_dict[key]["img_num"])])
        y_pred.append(predict)
        
    score_list = pd.DataFrame(score_list)
    preid_list = pd.DataFrame(preid_list)
    
    test_acc = correct / total

    return test_acc, y_true, y_pred, preid_list, score_list

def tabulate_loss (p_loss):
    loss_list = {}
    for key in p_loss.keys():
        loss_list[key] = p_loss[key]['img_loss']/p_loss[key]["img_num"]
    #loss_list = pd.DataFrame(loss_list)
    return pd.DataFrame(loss_list,index=[0])


def compute_grad(model,sample, clinic, target,loss_fn):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    #clinic = clinic.unsqueeze(0)
    target = target.unsqueeze(0)

    prediction = model(sample,clinic)
    loss = loss_fn(prediction, target)
    grad_list = []
    for l in loss:
      grad_list[l]=torch.autograd.grad(l,list(model.parameters()),allow_unused=True)
    return grad_list


def compute_sample_grads(model,data, clinic,targets,loss_fn,bs):
    """ manually process each sample with per sample gradient """
    sample_grads = [compute_grad(model,data[i], clinic,targets[i],loss_fn) for i in range(bs)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads


def train_model(model_ft, train_file, clin_features, clin_pts, pt_wts,num_epochs=10,
                batch_size = 64, num_workers = 1, lr = 0.001, momentum = 0.9, step_size=7, gamma=0.1):


    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainloader = torch.utils.data.DataLoader(
        CustomDset(train_file,  data_transforms),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    model_ft.to(device)
    #print(pt_wts.keys())
    model_ft.train()
    #added reduction='none' so we have per sample loss 
    criterion = nn.CrossEntropyLoss(reduction='none')  
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)  
    # Decay LR by a factor of 0.1 every 7 epochs    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 

    best_model_wts = copy.deepcopy(model_ft.state_dict())  
    best_acc = 0.0 
    
    steps = num_epochs * len(trainloader)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  

        #running_loss = 0.0
        running_loss = torch.zeros(1,dtype=torch.float64)
        running_loss = running_loss.to(device)
        running_corrects = 0
        total = 0

        person_prob_dict = dict()
        person_loss_dict = dict()

        # Iterate over data
        for inputs_, labels_, names_, _ in trainloader:
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
               
                X_train_minmax = torch.from_numpy(np.array([clin_features[:,clin_pts.index(i)] for i in names_], dtype=np.float32))
                batch_pt_wts = [pt_wts[k] for k in names_]
                batch_pt_wts = torch.from_numpy(np.array(list(batch_pt_wts)))
                batch_pt_wts = batch_pt_wts.to(device)
                #print(batch_pt_wts)
                outputs_ = model_ft(inputs_, X_train_minmax.to(device))

                process_probs(outputs_, names_, labels_, person_prob_dict,pt_wts)

                _, preds = torch.max(outputs_, 1)
                #loss with be per sample loss 
                loss = criterion(outputs_, labels_)
                #per_sample_grads = compute_sample_grads(model_ft,inputs_, X_train_minmax.to(device), labels_,criterioni,batch_size,names_)
                process_loss(loss, names_,person_loss_dict,pt_wts)
                #get mean of loss
                loss = loss * batch_pt_wts
                # backward + optimize only if in training phase
                #loss.backward()
                loss.mean().backward()
                optimizer.step()

            # statistics
            #running_loss += loss.item() * inputs_.size(0)
            loss = loss.to(device)
            #running_loss += loss * inputs_.size(0)
            running_loss = torch.cat((running_loss,loss))
            running_corrects += torch.sum((preds == labels_.data).int())
            
            total += preds.size(0)

        #epoch_loss = running_loss / total
        epoch_loss = running_loss
        epoch_acc = running_corrects / total

        epoch_acc_sample, *_ = tabulate_probs(person_prob_dict)
        epoch_loss_sample = tabulate_loss(person_loss_dict)
        scheduler.step()
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model_ft.state_dict())

        #print('{} Loss: {:.4f} Per Image Acc: {:.4f} Per Sample Acc: {:.4f}'.format(
        print('{} Loss: {} Per Image Acc: {:.4f} Per Sample Acc: {:.4f}'.format(
#            'train', epoch_loss, epoch_acc, epoch_acc_sample))
            'train', epoch_loss_sample, epoch_acc, epoch_acc_sample))
    # load best model weights
    model_ft.load_state_dict(best_model_wts)
    
    return model_ft, best_acc, steps

def test_model (model, test_file, clin_features, clin_pts, clin_wts,batch_size = 64, num_workers = 1):

    #test-specific transforms
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
   
    model.to(device)
    model.eval()
 
    testset = CustomDset(test_file, data_transforms)  
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                             shuffle = False, num_workers = num_workers)

    person_prob_dict = dict()
    with torch.no_grad():
        for data in testloader:
            images, labels, names_, images_names = data
            X_test_minmax = [clin_features[:,clin_pts.index(i)] for i in names_]
            outputs = model(images.to(device), torch.from_numpy(np.array(X_test_minmax, dtype=np.float32)).to(device))
            process_probs(outputs, names_, labels, person_prob_dict,pt_wts)
            
    test_acc, y_true, y_pred, preid_list, score_list = tabulate_probs(person_prob_dict)

    return test_acc, y_true, y_pred, preid_list, score_list

    

def preprocess_clinical (clinical_file):
    #note this is computing the min-max transform over all the data
    clin_dt=pd.read_csv(clinical_file)  
    clin_pts=[i for i in clin_dt.TCGA_ID]
    clin_wts=[i for i in clin_dt.weight]
    raw_clin_features=[clin_dt[i] for i in clin_dt.columns[1:(clin_dt.shape[1]-1)]]      
    min_max_scaler = preprocessing.MinMaxScaler()  
    clin_features = min_max_scaler.fit_transform(raw_clin_features)
    pt_wts = {}
    for key in range(len(clin_pts)):
       pt_wts[clin_pts[key]] = clin_wts[key]
    return clin_features, clin_pts, pt_wts

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_file", type=str, default="MMDL/label/clinic4_35.csv", nargs="?")
    parser.add_argument("--train_file", type=str, default="data/train_0.csv")
    parser.add_argument("--test_file", type=str, default="data/test_0.csv")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--n_epochs", type=int, default="10")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    clin_features, clin_pts, pt_wts= preprocess_clinical(args.clinical_file)

    since = time.time()

    #can tweak model here if needed 
    model = Cnn_With_Clinical_Net(clin_features.shape[0]) 

    model_trn, train_acc, steps = train_model(model, args.train_file, clin_features, clin_pts, pt_wts, num_epochs=args.n_epochs)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m'.format(
        time_elapsed // 3600, (time_elapsed-time_elapsed // 3600) * 60))
    print('Best train Acc: {:4f}'.format(train_acc))

    trn_basename = os.path.basename(args.train_file)

    torch.save(model_trn, args.out_dir+f'/model_fit_{trn_basename}.pkl' )

    print('Starting testing...')

    test_acc, y_true, y_pred, preid_list, score_list = test_model (model_trn, args.test_file, clin_features, clin_pts, pt_wts)

    test_basename = os.path.basename(args.test_file)

    np.save(args.out_dir+f'/y_true_{test_basename}.npy', np.array(y_true)) 
    np.save(args.out_dir+f'/preid_{test_basename}.npy', np.array(preid_list))
    np.save(args.out_dir+f'/score_{test_basename}.npy', np.array(score_list))
    np.save(args.out_dir+f'/y_pred_{test_basename}.npy', np.array(y_pred))
    
    print('Testing complete...')
    print ('Accuracy of the network on test images: %d %%' % (
        100 * test_acc))

        
