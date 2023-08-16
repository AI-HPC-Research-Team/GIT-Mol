# -*- coding: utf-8 -*-
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
from peft import get_peft_model
from transformers import T5Tokenizer, T5ForConditionalGeneration

from momu import MoMu
import rdkit.Chem as Chem

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.blip2.GITformer import BertConfig, BertLMHeadModel


class SelfAttentionFusion(nn.Module):
    def __init__(self, d_model, nhead, num_layers, pool):
        super(SelfAttentionFusion, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.pool = pool
    
    def forward(self, features_text, features_graph):
       
        features_text = features_text.transpose(0, 1)
        features_graph = features_graph.transpose(0, 1)

        sequence = torch.cat([features_text, features_graph], dim=0)
        #print(f"sequence.shape : {sequence.shape}")

        output = self.transformer_encoder(sequence)

        output = output.transpose(0, 1)
        output = output.transpose(1, 2)

        if(self.pool == 'avg'):
            features = F.avg_pool1d(output, output.shape[2]).squeeze(2)
        else:
            features = F.max_pool1d(output, output.shape[2]).squeeze(2)
        return features

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # Add dropout after the first hidden layer
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)
        
class FusionNet(nn.Module):
    def __init__(self, input_size):
        super(FusionNet, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.weights = 0.5
    def forward(self, features_text, features_graph):
        combined = torch.cat([features_text, features_graph], dim=-1)
        self.weights = torch.sigmoid(self.fc(combined))
        fused_embedding = self.weights * features_text + (1 - self.weights) * features_graph
        return fused_embedding
    
class LossWeights(nn.Module):
    def __init__(self, num_losses):
        super(LossWeights, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_losses)) # 初始化权重

    def forward(self):
        return F.softmax(self.weights, dim=0)
        
class GITModel(nn.Module):
    def __init__(self, config = None, fp = False, task = None, device = None, mode = None, num_tasks = None, pool = None, fusion_mode = None):
        super().__init__()

        self.blip2conf = Blip2Config()
        self.model = Blip2ForConditionalGeneration(self.blip2conf)
        self.model.vision_model = None

        self.model.language_model = T5ForConditionalGeneration.from_pretrained("../../../molt5-large")
        self.model.language_projection = nn.Linear(768, 1024)

        language_lora_config = LoraConfig(
            peft_type="LORA",
            r=16,  # rank of the update matrices
            lora_alpha=16,  # scaling factor for LoRA updates
            target_modules=["q","v","lm_head","shared"],  # apply LoRA to query and value matrices
            lora_dropout=0.1,  # dropout rate for LoRA updates
            bias="none",  # do not train bias parameters
                #modules_to_save=["lm_head"]  # also train the classifier parameters
        )
        self.model.language_model  = get_peft_model(self.model.language_model, language_lora_config)

        
        self.model.graph_encoder = GraphEncoder(config)
        self.device = device
        self.temperature = nn.Parameter(torch.tensor(0.01)) 
        self.property_projection_graph = nn.Linear(300, 768)
        self.property_projection_text = nn.Linear(1024, 768)
        self.num_tasks = num_tasks
        self.property_mlp = MLP(768,256,self.num_tasks)
        self.mode = mode
        self.inputs_modal = task
        self.fusionnet = FusionNet(768*2)
        self.pool = pool
        self.fusion_model = SelfAttentionFusion(d_model=768, nhead=8, num_layers=1,pool = 'avg')
        self.fusion_mode = fusion_mode
       
    
    def get_embeddings_outputs(self, mol, inputs_modal):
        language_model_inputs_text = None
        language_model_inputs_graph = None
        input_tensors = []
        if(self.fusion_mode == 'attention'):
            text = mol['SMILES']
            smiles_embeddings = self.model.language_model.encoder(**text).last_hidden_state
            features_text = self.property_projection_text(smiles_embeddings)
            graph_embeddings, graph_attention_mask = self.model.graph_encoder.encode_graph2d(mol)
            features_graph = self.property_projection_graph(graph_embeddings)
            embeddings_outputs = self.fusion_model(features_text, features_graph)
        else:
            if 'SMILES' in inputs_modal:
                text = mol['SMILES']
                #print(f"text : {text}")
                smiles_embeddings = self.model.language_model.encoder(**text).last_hidden_state
                
                smiles_embeddings = smiles_embeddings.transpose(1, 2) 
                if(self.pool == 'avg'):
                    features_text = F.avg_pool1d(smiles_embeddings, smiles_embeddings.shape[2]).squeeze(2)
                else:
                    features_text = F.max_pool1d(smiles_embeddings, smiles_embeddings.shape[2]).squeeze(2)
                features_text = self.property_projection_text(features_text)
                input_tensors.append(features_text)
        
            if 'graph2d' in inputs_modal:
                graph_embeddings, graph_attention_mask = self.model.graph_encoder.encode_graph2d(mol)
                
                graph_embeddings = graph_embeddings.transpose(1, 2) 
                if(self.pool == 'avg'):
                    features_graph = F.avg_pool1d(graph_embeddings, graph_embeddings.shape[2]).squeeze(2)
                else:
                    features_graph = F.max_pool1d(graph_embeddings, graph_embeddings.shape[2]).squeeze(2)
                features_graph = self.property_projection_graph(features_graph)
                input_tensors.append(features_graph)
                
            if 'SMILES' in inputs_modal and 'graph2d' in inputs_modal and self.fusion_mode == 'weight':
                embeddings_outputs = self.fusionnet(features_text, features_graph)
            else:
                embeddings_outputs = torch.stack(input_tensors).mean(dim=0)
        return embeddings_outputs
    
    def forward(self, mol):
        embeddings = self.get_embeddings_outputs(mol, self.inputs_modal)
        result = self.property_mlp(embeddings)
        return result
    

 
class GraphEncoder(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        self.graph2d_encoder = MoMu(config["graph"]).graph_encoder

    def forward(self, graph2ds, graph3ds):
        graph_loss = None
        return graph_loss
    def encode_graph2d(self, mol):
        _, node_feats, node_feats_mask = self.graph2d_encoder(mol["graph2d"])
        return node_feats ,node_feats_mask
