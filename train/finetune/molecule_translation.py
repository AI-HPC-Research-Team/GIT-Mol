# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
from dataset.dataset import gitDataset

from utils import AverageMeter, ToDevice, print_model_info
from models.model_finetune import GITModel
from accelerate import Accelerator
import datetime
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch

from transformers import T5ForConditionalGeneration, BertTokenizerFast
from evaluations.text_translation_metrics import text_evaluate
from evaluations.mol_translation_metrics import mol_evaluate
from evaluations.fingerprint_metrics import molfinger_evaluate

from torch.optim.lr_scheduler import StepLR

def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    for key in elem_keys:
        if key in ['smiles_labels','caption_labels','isosmiles_labels']:
            input_ids_tensors = [elem[key] for elem in batch]
            input_ids_tensors = [tensor.squeeze(0) for tensor in input_ids_tensors]
            padded_input_ids = pad_sequence(input_ids_tensors, batch_first=True)
            collated_batch[key] = padded_input_ids
        elif(key in ['graph2d','graph3d']):
            molecule_batch = Batch.from_data_list([elem[key] for elem in batch])
            collated_batch[key] = molecule_batch
        elif key in ['smiles','cid','caption','isosmiles']:
            collated_batch[key] = [item[key] for item in batch]
        elif key in ['SMILES','Caption','isoSMILES']:
            input_ids_tensors = [elem[key].input_ids for elem in batch]
            input_ids_tensors = [tensor.squeeze(0) for tensor in input_ids_tensors]
            padded_input_ids = pad_sequence(input_ids_tensors, batch_first=True)
            attention_mask_tensors = [elem[key].attention_mask for elem in batch]
            attention_mask_tensors = [tensor.squeeze(0) for tensor in attention_mask_tensors]              
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

def train_git_decoder(train_loader, val_loader, test_loader, model, optimizer, scheduler, args, device, task, best_loss = None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss": [], "val_loss": [], "test_loss": []}
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        
        #model.train()
        train_loss = []
        train_loader = tqdm(train_loader, desc="Training")
        for mol in train_loader:
            smiles = mol['smiles']
            if "cid" in mol:
                del mol["cid"]
            if "smiles" in mol:
                del mol["smiles"]
            if "isosmiles" in mol:
                del mol["isosmiles"]
            if "caption" in mol:
                del mol["caption"]
            mol["smiles"] = smiles
            mol = ToDevice(mol, device)
            loss = model(mol)
            accelerator.backward(loss)
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()


            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                train_loss.append(running_loss.get_average())
                running_loss.reset()

        loss_values["train_loss"].append(np.mean(train_loss))
        val_loss = val_git(val_loader, model, task, device)
        test_loss = val_git(test_loader, model, task, device)
        loss_values["val_loss"].append(val_loss)
        loss_values["test_loss"].append(test_loss)
        if best_loss == None or val_loss<best_loss :
        #if True:
            best_loss = val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            
            torch.save({
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'best_loss': best_loss
            }, os.path.join(args.output_path, f"checkpoint_{epoch}_{timestamp}.pth"))
          
            message = f"best_loss:{best_loss} ,val_loss:{val_loss}, checkpoint_{epoch}_{timestamp}.pth saved"
            print(message)
            # Write the message to the file
            with open(args.result_save_path, 'a') as f:   
                f.write(message + "\n")   
        else:
            message = f"best_loss:{best_loss} ,val_loss:{val_loss}, ckpt passed"
            print(message)
            with open(args.result_save_path, 'a') as f:   
                f.write(message + "\n") 
        print(loss_values)


def get_latest_checkpoint(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        return None

    all_checkpoints = [filename for filename in os.listdir(checkpoints_dir) if filename.startswith("checkpoint_")]

    if not all_checkpoints:
        return None

    latest_checkpoint = max(all_checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
    return os.path.join(checkpoints_dir, latest_checkpoint)

def test_smiles(cid_list, targets, preds, descriptions, inputs_modal):
    bleu_score, exact_match_score, levenshtein_score, validity_score, result_dataframe = mol_evaluate(targets, preds, descriptions)
    finger_metrics = molfinger_evaluate(targets, preds)

    message = "input: {}, Metrics: bleu_score:{}, em-score:{}, levenshtein:{}, maccs fts:{}, rdk fts:{}, morgan fts:{}, validity_score:{}".format(inputs_modal, bleu_score, exact_match_score, levenshtein_score, finger_metrics[1], finger_metrics[2], finger_metrics[3], validity_score)
    print(message)
    print(result_dataframe.head())
    with open(args.smiles_save_path, 'a') as f:
        result_dataframe.to_csv(f, header=f.tell()==0, sep='\t', index=False)
    return message

def test_caption(cid_list, targets, preds, smiles, inputs_modal):

    tokenizer = BertTokenizerFast.from_pretrained("ckpts/text_ckpts/scibert_scivocab_uncased")
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score, result_dataframe = text_evaluate(tokenizer, targets, preds, smiles, 512)
    message = 'input: {}, Metrics: bleu-2:{}, bleu-4:{}, rouge-1:{}, rouge-2:{}, rouge-l:{}, meteor-score:{}'.format(inputs_modal, bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score)
    print(message)
    with open(args.caption_save_path, 'a') as f:
        result_dataframe.to_csv(f, header=f.tell()==0, sep='\t', index=False)
    return message

def val_git(val_loader, model, task_list, device):
    model.eval()
    val_loss = 0
    logger.info("Validating...")
    with torch.no_grad():
        val_loader = tqdm(val_loader, desc="Validation")
        for i, mol in enumerate(val_loader):
            if "cid" in mol:
                del mol["cid"]
            if "smiles" in mol:
                smiles = mol["smiles"]
                del mol["smiles"]
            if "isosmiles" in mol:
                isosmiles = mol["isosmiles"]
                del mol["isosmiles"]
            if "caption" in mol:
                caption = mol["caption"]
                del mol["caption"]
            mol = ToDevice(mol, device)
            mol['smiles'] = smiles
            loss = model(mol)
            if(i==1):
                print(f"batch_{i}, device: {device}")
                for task in task_list:
                    print(f"task : {task}")
                    inputs_modal = task['inputs_modal']
                    outputs_modal = task['outputs_modal']
                    if('SMILES' in outputs_modal or 'isoSMILES' in outputs_modal):
                        result = model.generate_text(mol, inputs_modal, outputs_modal)
                        print(f"SMILES : {smiles[0]} , result : {result[0]}")
                    if('caption' in outputs_modal):
                        result = model.generate_text(mol, inputs_modal, outputs_modal)
                        print(f"caption : {caption[0]} , result : {result[0]}")
            val_loss += loss.detach().cpu().item()
        logger.info("validation loss %.4lf" % (val_loss / len(val_loader)))
    return val_loss / len(val_loader)

def test_git(test_loader, model, task_list, device):
    model.eval()
    test_loss = 0
    
    logger.info("Testing...")
    with torch.no_grad():
        test_loader = tqdm(test_loader, desc="Test")
        for task in task_list:
            print(f"task : {task}")
            cid_list = []
            smiles_list = []
            caption_list = []
            pre_smiles_list = []
            pre_caption_list = []
            for mol in test_loader:
                smiles = mol['smiles']
                #print(f"smiles 1: {smiles}")
                smiles_list=smiles_list+smiles
                if "smiles" in mol:
                    del mol["smiles"]
                caption = mol['caption']
                caption_list = caption_list+caption
                if "caption" in mol:
                    del mol["caption"]
                mol = ToDevice(mol, device)
                mol['smiles'] = smiles
                #loss = model(mol)

                inputs_modal = task['inputs_modal']
                outputs_modal = task['outputs_modal']
                if('isoSMILES' in outputs_modal):
                    pre_smiles = model.generate_text(mol, inputs_modal, outputs_modal)
                    print(f"isoSMILES : {smiles[0]} , result : {pre_smiles[0]}")
                    pre_smiles_list = pre_smiles_list+pre_smiles
                if('caption' in outputs_modal):
                    pre_caption = model.generate_text(mol, inputs_modal, outputs_modal)
                    print(f"caption : {caption[0]} , result : {pre_caption[0]}")
                    pre_caption_list = pre_caption_list+pre_caption
                    
                #test_loss += loss.detach().cpu().item()
            #print(f"test_loss : {test_loss}")

            if('isoSMILES' in outputs_modal):
                assert len(smiles_list) == len(pre_smiles_list), "Lists must have the same length."
                result = test_smiles(cid_list, smiles_list, pre_smiles_list, caption_list, inputs_modal)
            elif('caption' in outputs_modal):
                assert len(caption_list) == len(pre_caption_list), "Lists must have the same length."
                result = test_caption(cid_list, caption_list, pre_caption_list, smiles_list, inputs_modal)

def add_arguments(parser):
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_path", type=str, default="configs/git.json")
    parser.add_argument("--output_path", type=str, default="ckpts/finetune_ckpts/git_decoder")
    parser.add_argument("--caption_save_path", type=str, default="../../assets/smiles2caption/test.csv")
    parser.add_argument("--smiles_save_path", type=str, default="../../assets/image2smiles/test.csv")
    parser.add_argument("--result_save_path", type=str, default="../../assets/git_decoder/result.csv")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=200)
    parser.add_argument("--task", type=str, default="molt5")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    # image2d:1
    # isoSMILES:4 ,caption:5
    # graph2d:6
    task1_4 = {
        'inputs_modal': ['image2d'],
        'outputs_modal': ['isoSMILES']
    }
    task1_5 = {
        'inputs_modal': ['image2d'],
        'outputs_modal': ['caption']
    }
    task6_5 = {
        'inputs_modal': ['graph2d'],
        'outputs_modal': ['caption']
    }
    task4_5 = {
        'inputs_modal': ['isoSMILES'],
        'outputs_modal': ['caption']
    }
    task5_4 = {
        'inputs_modal': ['caption'],
        'outputs_modal': ['isoSMILES']
    }
    task64_5 = {
        'inputs_modal': ['graph2d','isoSMILES'],
        'outputs_modal': ['caption']
    }
    # load dataset
    if(args.task == 'molt5'):
        train_data_file = "../../igcdata/molcap/train_72k.pkl"
        val_data_file = "../../igcdata/molcap/valid_9k.pkl"
        test_data_file = "../../igcdata/molcap/test_9k.pkl"
        task = [task64_5]
        #task = [task4_5]
        #task = [task6_5]

        
        #train_data_file = "../../igcdata/chebi-20/train.txt"
        #val_data_file = "../../igcdata/chebi-20/validation.txt"
        #test_data_file = "../../igcdata/chebi-20/test.txt"
        #task = [task5_4]

    elif(args.task == 'swin_molt5'):
        train_data_file = "../../igcdata/molcap/train_72k.pkl"
        val_data_file = "../../igcdata/molcap/valid_9k.pkl"
        test_data_file = "../../igcdata/molcap/test_9k.pkl"
        task = [task1_5]
        #task = [task1_4]
        
    print(f"task : {task}")
    
    latest_checkpoint = get_latest_checkpoint(args.output_path)

    if latest_checkpoint:
        print(f"Latest checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found.")
    
    if args.mode == "train":
        logger.info(f"mode : {args.mode} ")
        accelerator = Accelerator()
        logger.info("Loading model ......")

        
        model = GITModel(config = config['network'], task = task, device = accelerator.device)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = False)
            #model.load_state_dict(state_dict, strict = True)
            
        #model.model.language_model = T5ForConditionalGeneration.from_pretrained("../../ckpts/text_ckpts/molt5-base")
        
        best_loss = None
        
        print_model_info(model,level=2)
        logger.info("Loading model successed")
        
        logger.info("Loading dataset ......")
        train_dataset = gitDataset(train_data_file, config["data"]["drug"], split="train", task = task)
        val_dataset = gitDataset(val_data_file, config["data"]["drug"], split="val", task = task)
        test_dataset = gitDataset(test_data_file, config["data"]["drug"], split="test", task = task)
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)

        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, 4, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")
        
        
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        device = accelerator.device
        model = model.to(device)
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
        val_loader = accelerator.prepare_data_loader(val_loader, device_placement=True)
        test_loader = accelerator.prepare_data_loader(test_loader, device_placement=True)
        train_git_decoder(train_loader, val_loader, test_loader, model, optimizer, scheduler, args, device, task, best_loss)
        
    elif args.mode == "test":
        logger.info(f"mode : {args.mode} ")
        logger.info("Loading dataset ......")
        test_dataset = gitDataset(test_data_file, config["data"]["drug"], split="test", task = task)
        logger.info("Loading dataset successed")
        
        logger.info("Loading dataloader ......")
        test_loader = DataLoader(test_dataset, 12, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")
        
        logger.info("Loading model ......")
        accelerator = Accelerator()
        device = accelerator.device
        model = GITModel(config = config['network'], task = task, device = device)
        if latest_checkpoint is not None :
            state_dict = torch.load(latest_checkpoint, map_location='cpu')["model_state_dict"]
            best_loss = torch.load(latest_checkpoint, map_location='cpu')["best_loss"]
            model.load_state_dict(state_dict, strict = False)

        print_model_info(model,level=2)
        logger.info("Loading model successed")
        model = model.to(device)
        test_loader = accelerator.prepare_data_loader(test_loader, device_placement=True)
        test_git(test_loader, model, task, device)
        
    elif args.mode == "datacheck":
        logger.info(f"mode : {args.mode} ")
        test_dataset = gitDataset(val_data_file, config["data"]["drug"], split="val", task = task)
        for i in range(2):
            print(test_dataset[i])
        print(f"len(test_dataset): {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
        for i, batch in enumerate(test_loader):
            if i >= 2:
                break
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"Shape of {key}: {value.shape}")
                else:
                    print(f"{key} is not a tensor")
                    print(f"{key} : {value}")
                    
    elif args.mode == "modelcheck":
        logger.info(f"mode : {args.mode} ")
        test_dataset = gitDataset(val_data_file, config["data"]["drug"], split="val", task = task)
        model = GITModel(config = config['network'], task = task, device = args.device)
        model.to(args.device)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,pin_memory=True)
        for i, mol in enumerate(test_loader):
            if i >= 2:
                break     
            smiles = mol["smiles"]
            if "smiles" in mol:
                del mol["smiles"]
            if "caption" in mol:
                del mol["caption"]
            mol["smiles"] = smiles
            mol = ToDevice(mol, args.device)
            output = model(mol)
            print(output)
