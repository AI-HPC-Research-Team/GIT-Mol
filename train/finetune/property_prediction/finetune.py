import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

from loader import MoleculeDataset
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import random

from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd
import os
import shutil

from utils.utils import print_model_info, EarlyStopping, ToDevice
from model import GITModel

from torch.optim.lr_scheduler import StepLR

#criterion = nn.BCEWithLogitsLoss(reduction = "none")
criterion = nn.BCEWithLogitsLoss(reduction = "mean")

def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    
    for key in elem_keys:
        if(key in ['graph2d']):
            molecule_batch = Batch.from_data_list([elem[key] for elem in batch])
            collated_batch[key] = molecule_batch
        elif key in ['SMILES']:
            input_ids_tensors = [torch.tensor(elem[key].input_ids) for elem in batch]
            padded_input_ids = pad_sequence(input_ids_tensors, batch_first=True)
            attention_mask_tensors = [torch.tensor(elem[key].attention_mask) for elem in batch]
            padded_attention_mask = pad_sequence(attention_mask_tensors, batch_first=True)
            collated_batch[key] = {
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask
            }
        else:
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data
                

    return collated_batch
    
def train(args, epoch, model, device, loader, optimizer):
    model.train()
    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = ToDevice(batch, device)
        pred = model(batch)
        y = batch['graph2d'].y.view(pred.shape).to(torch.float64)

      
        is_valid = y**2 > 0

     
        y = torch.where(is_valid, y, torch.zeros(y.shape).to(y.device).to(y.dtype))

        loss = criterion(pred.double(), (y+1)/2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f}")
        
def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        #batch = batch.to(device)
        batch = ToDevice(batch, device)
        with torch.no_grad():
            #pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model(batch)

        y_true.append(batch['graph2d'].y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'sider', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'mola', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--mode', type=str, default = 'train', help='mode in [train, test, datacheck]')
    #parser.add_argument('--model_mode', type=str, default = None, help='use qformer')
    parser.add_argument('--config_path', type=str, default = 'config.json', help='config of model')
    parser.add_argument('--modals', type=str, default = None, help='config of model')  
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ckpt_output_root", type=str, default = '/userhome/mola_ckpts/finetune')
    parser.add_argument("--pool", type=str, default = 'avg')
    parser.add_argument("--fusion_mode", type=str, default = 'attention')
    parser.add_argument("--ckpt", type=str, default = None)
    args = parser.parse_args()
    
    
    #args.runseed = random.randint(0, 10)
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    print("Loading dataset ......")
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)
    if(args.modals == None):
        task = ['graph2d','SMILES']
    else:
        task = [args.modals]
    
    #print(f"task : {task}")
    print(f"runseed : {args.runseed}")
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")
        
    print("Loading dataset successed")
    print('++++++++++', train_dataset[0])
    print("Loading dataloader ......")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=custom_collate_fn)
    print("Loading dataloader successed")
    
    if(args.mode == 'train'):
        config = json.load(open(args.config_path))

        print("Loading model ......")

        latest_checkpoint = 'ckpts/pt-gs_avg_max_100_3.pth'
        if(latest_checkpoint == None):
            print(f"None input_ckpt , use default")
           
        model = GITModel(config = config['network'], task = task, device = device, num_tasks = num_tasks, mode = args.model_mode,pool = args.pool,fusion_mode = args.fusion_mode)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            #best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = False)
        print_model_info(model,level=2)
        print("Loading model successed")
        
        model.to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        print(optimizer)

        train_acc_list = []
        val_acc_list = []
        test_acc_list = []

            
        stopper_mode = "higher"
        ckpt_root = f"{args.ckpt_output_root}/{args.dataset}"
        if(os.path.exists(ckpt_root)):
            pass
        else:
            os.makedirs(ckpt_root)
            
        if(args.model_mode == None):
            if(args.modals == None):
                args.ckpt_output_path = f"{ckpt_root}/{args.dataset}"
            else:
                args.ckpt_output_path = f"{ckpt_root}/{args.dataset}_{args.modals}"
        else:
            if(args.modals == None):
                args.ckpt_output_path = f"{ckpt_root}/{args.dataset}_{args.model_mode}"
            else:
                args.ckpt_output_path = f"{ckpt_root}/{args.dataset}_{args.modals}_{args.model_mode}"
                
        stopper = EarlyStopping(
            mode=stopper_mode, patience=args.patience, filename=args.ckpt_output_path)
        if(args.modals == None):
            exp_path = '/userhome/{}_results/{}_{}/'.format(args.input_model_file, args.dataset, args.model_mode)
        else:
            exp_path = '/userhome/{}_results/{}_{}/{}/'.format(args.input_model_file, args.dataset, args.model_mode, args.modals)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        ckpt_name = ""
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))

            model.train()
            epoch_iter = tqdm(train_loader, desc="Iteration")
            for step, batch in enumerate(epoch_iter):
                batch = ToDevice(batch, device)
                pred = model(batch)
                y = batch['graph2d'].y.view(pred.shape).to(torch.float64)
                is_valid = y**2 > 0
                y = torch.where(is_valid, y, torch.zeros(y.shape).to(y.device).to(y.dtype))
        
                loss = criterion(pred.double(), (y+1)/2)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f}")
            scheduler.step()

            print("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_acc = 0
            val_acc  = eval(args, model, device, val_loader)
            test_acc = eval(args, model, device, test_loader)

            print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_acc_list.append(train_acc)

                
            early_stop, avg_score, std_dev ,ckpt_name = stopper.step((val_acc), model)
                             
            result_file_path = f"{exp_path}result.txt"
            if not os.path.isfile(result_file_path):
             
                open(result_file_path, 'w').close()
            if(early_stop):
                with open(result_file_path, "a") as result_file:
                    result_file.write(f"{args.ckpt_output_path} / {ckpt_name}\n")
                    result_file.write(f" val: {val_acc*100:.2f}%, test: {test_acc*100:.2f}%, Average score: {avg_score*100:.2f}% ± {std_dev*100:.2f}% , ckpt: {latest_checkpoint}, pool : {args.pool}, fusion_mode : {args.fusion_mode}, lr:{args.lr} \n") 
                break
        """
        with open(result_file_path, "a") as result_file:
            result_file.write(f"{args.ckpt_output_path} \n")
            result_file.write(f" val: {val_acc*100:.2f}%, test: {test_acc*100:.2f}%, Average score: {avg_score*100:.2f}% ± {std_dev*100:.2f}% , ckpt: {latest_checkpoint}, pool : {args.pool}\n") 
        """
        print(f'runseed: {args.runseed}' )
        print('Best val_epoch:', val_acc_list.index(max(val_acc_list)))
        print('Best val_test_auc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])
        print('Best test_epoch:', test_acc_list.index(max(test_acc_list)))
        print('Best test_auc:', max(test_acc_list))
                                                          



        df = pd.DataFrame({'train':train_acc_list,'valid':val_acc_list,'test':test_acc_list})
        df.to_csv(exp_path + 'seed{}.csv'.format(args.runseed))

        logs = 'Dataset:{}, runseed:{}, dataseed: {} , Best val_epoch:{}, Best val_test_auc:{:.5f}, Best test_epoch:{}, Best test_auc:{:.5f}'.format(
            args.dataset, args.runseed, args.seed,  val_acc_list.index(max(val_acc_list)), 
            test_acc_list[val_acc_list.index(max(val_acc_list))],
            test_acc_list.index(max(test_acc_list)),
            max(test_acc_list)
        )
        logs = logs + f" ckpt: {latest_checkpoint}, pool : {args.pool}, fusion_mode : {args.fusion_mode}, lr:{args.lr} ,patience : {args.patience}, model_ckpt : {ckpt_name}, batch_size : {args.batch_size}"
        with open(exp_path + '{}_log.csv'.format(args.dataset),'a+') as f:
            f.write('\n')
            f.write(logs)
    elif(args.mode =='eval'):
        config = json.load(open(args.config_path))

        print("Loading model ......")

        if(args.ckpt == None):
            print(f"None ckpt")
            latest_checkpoint = None
        else:
            latest_checkpoint = args.ckpt
            print(f"ckpt:{latest_checkpoint}")
        
        model = GITModel(config = config['network'], task = task, device = device, num_tasks = num_tasks, mode = args.model_mode,pool = args.pool,fusion_mode = args.fusion_mode)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            #best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict)
        print_model_info(model,level=2)
        print("Loading model successed")
        
        model.to(device)
        test_acc_list = []
        for i in range(3):
            test_acc = eval(args, model, device, test_loader)
            test_acc_list.append(test_acc)
        print(f"test_acc_list : {test_acc_list}")
        #print(f"avg_test_acc : {np.mean(test_acc_list)}")

    elif(args.mode =='datacheck'):
        print(f"dataset == {args.dataset}, num_tasks = {num_tasks}")
        for i in range(2):
            data = valid_dataset[i]
            print(f"data {i} : {data}")
                  
        for i, batch in enumerate(val_loader):
            if i >= 2:
                break
            print(f"batch {i} : {batch}")
                  
if __name__ == "__main__":
    main()
