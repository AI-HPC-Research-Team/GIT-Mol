# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from graph_featurizer import GraphFeaturizer


from utils.utils import valid_smiles
import pandas as pd
from PIL import Image
from transformers import BertTokenizer, T5Tokenizer
from transformers import Blip2Processor
from torchvision import transforms
import torch.nn as nn
import re
import numpy as np
import random
from itertools import combinations
from rdkit.Chem.rdchem import HybridizationType, BondType
from torch_geometric.data import Data


BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

class BaseDataset(Dataset): #, ABC):
    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self._load_data()
        #self._featurize()
        #self._load_mols()
        
    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.smiles)

class gitDataset(BaseDataset):
    def __init__(self, data_path, config, split, encoder = False, task = None, transform=None):
        if data_path.endswith('.pkl'):
            self.data = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.txt'):
            self.data = pd.read_table(data_path)
        else:
            raise ValueError(f'Unsupported file extension in: {data_path}')
            
        self.split = split
        self.encoder = encoder
        self.processor = Blip2Processor.from_pretrained("../../ckpts/fusion_ckpts/blip2")
        self.label_tokenizer = T5Tokenizer.from_pretrained("../../ckpts/text_ckpts/molt5-base",  model_max_length=512)
        self.tokenizer = BertTokenizer.from_pretrained("../../ckpts/text_ckpts/scibert_scivocab_uncased", truncation_side='right', model_max_length=512)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.processor = Blip2Processor(self.processor.current_processor, self.label_tokenizer)
        #graph_feature
        graph2d_featurizer_config = { 
            'name' : 'ogb'
        }
        self.task_list = task

        self.graph2d_featurizer = DrugGraphFeaturizer(graph2d_featurizer_config)
        if(self.split == 'train'):
            self.transform = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                # transforms.RandomErasing(p=0.3,scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        super(gitDataset, self).__init__(config)
                                               #for bond in mol.GetBonds())))
    def _load_data(self):
        self.smiles = []
        self.isosmiles = []
        self.captions = []
        self.cids = []
        self.image2d = []
        self.graph2d = []
        self.inputs_modal = []
        self.outputs_modal = []

        for task in self.task_list:
            inputs_modal = task['inputs_modal']
            self.inputs_modal = self.inputs_modal+inputs_modal
            outputs_modal = task['outputs_modal']
            self.outputs_modal = self.outputs_modal+outputs_modal
 
        self.inputs_modal = list(set(self.inputs_modal))  
        self.outputs_modal = list(set(self.outputs_modal)) 
        self.modality = self.inputs_modal + self.outputs_modal
        #print(self.inputs_modal)
        for _, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            smiles = row["isosmiles"]
            if valid_smiles(smiles):
                self.captions.append(row["summary"])
                self.cids.append(row["cid"])
                if "SMILES" in self.modality:
                    smiles = row["smiles"]
                    self.smiles.append(row["smiles"])
                else:
                    smiles = row["isosmiles"]
                    self.smiles.append(row["isosmiles"])
                if "isoSMILES" in self.modality:
                    self.isosmiles.append(row["isosmiles"])

                if "image2d" in self.modality:
                    img_file2d = '../../igcdata/image2d_resized/'+ row["image2d"]
                    #img_file2d = '../../'+ row["image_path"]
                    img2d = Image.open(img_file2d).convert('RGB')
                    img2d = self.transform(img2d)  
                    self.image2d.append(img2d)
                if "graph2d" in self.modality:
                    graph2d = self.graph2d_featurizer(smiles)
                    self.graph2d.append(graph2d)

            else:
                pass
    
    def __getitem__(self, i):
        inputs_1 = {}
        if "image2d" in self.inputs_modal:
            inputs_1["image2d"] = self.image2d[i]
            
        if('isoSMILES' in self.outputs_modal):
            isosmiles = self.isosmiles[i]
            isosmiles_labels = self.processor(text=isosmiles, return_tensors="pt")['input_ids']
            inputs_1['isosmiles_labels'] = isosmiles_labels
            
        if('SMILES' in self.outputs_modal):
            smiles = self.smiles[i]
            smiles_labels = self.processor(text=smiles, return_tensors="pt")['input_ids']
            inputs_1['smiles_labels'] = smiles_labels
            #inputs_1['smiles'] = smiles
        if('caption' in self.outputs_modal):
            caption = self.captions[i]
            caption_labels = self.processor(text=caption, return_tensors="pt")['input_ids']
            inputs_1['caption_labels'] = caption_labels
            inputs_1['caption'] = caption
            
        if('SMILES' in self.inputs_modal):
            #smiles = self.processor(text=self.smiles[i], return_tensors="pt")
            smiles = self.tokenizer(
                self.smiles[i],
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs_1['SMILES'] = smiles
        if('isoSMILES' in self.inputs_modal):
            #isosmiles = self.processor(text=self.isosmiles[i], return_tensors="pt")
            isosmiles = self.tokenizer(
                self.isosmiles[i],
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs_1['isoSMILES'] = isosmiles
            #inputs_1['isosmiles'] = self.isosmiles[i]
        if('caption' in self.inputs_modal):
            #caption = self.processor(text=self.captions[i], return_tensors="pt")
            caption = self.tokenizer(
                self.captions[i],
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs_1['Caption'] = caption
        if('graph2d' in self.inputs_modal):
            inputs_1['graph2d'] = self.graph2d[i]

        #inputs_1['cid'] = self.cids[i]
        inputs_1['smiles'] = self.smiles[i]
        inputs_1['caption'] = self.captions[i]
        
        return inputs_1