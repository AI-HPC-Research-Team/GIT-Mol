# -*- coding: utf-8 -*-
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer
from models.swin_transformer import SwinTransformer

from models.momu import MoMu
from models.GIT_former import BertConfig, BertLMHeadModel
from transformers.modeling_outputs import BaseModelOutput

import rdkit.Chem as Chem
import numpy as np

modalities = {
    'image2d': 'image',
    'SMILES': 'SMILES representation',
    'isoSMILES': 'isoSMILES representation',
    'caption': 'textual description',
    'graph2d': 'structure graph'
}
def generate_prompt(inputs, outputs):
    inputs_desc = [modalities[i] for i in inputs]
    outputs_desc = [modalities[o] for o in outputs]

    inputs_str = ' and '.join(inputs_desc)
    outputs_str = ', '.join(outputs_desc)
    
    prompt = f"Given the provided {inputs_str}, generate the corresponding {outputs_str}."
    #prompt = f"This is the {inputs_str} of the molecule, and the corresponding {outputs_str} is :"
    return prompt

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class GITFormer(nn.Module):
    def __init__(self, num_query_token, vision_graph_width, cross_attention_freq = 2):
        super().__init__()
        encoder_config = BertConfig.from_pretrained("../../ckpts/text_ckpts/scibert_scivocab_uncased")
        #encoder_config = BertConfig.from_pretrained("../../ckpts/text_ckpts/bert-base-uncased")
        encoder_config.encoder_width = vision_graph_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        self.Qformer = BertLMHeadModel.from_pretrained(
            "../../ckpts/text_ckpts/scibert_scivocab_uncased", config=encoder_config
        )
        """
        self.Qformer = BertLMHeadModel.from_pretrained(
            "../../ckpts/text_ckpts/bert-base-uncased", config=encoder_config
        )
        """
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)


class GITModel(nn.Module):
    def __init__(self, config = None, fp = False, task = None, device = None):
        super().__init__()

        self.blip2conf = Blip2Config()
        self.model = Blip2ForConditionalGeneration(self.blip2conf)
        self.model.language_model = T5ForConditionalGeneration.from_pretrained("../../ckpts/text_ckpts/molt5-base")
        
        self.processor = Blip2Processor.from_pretrained("../../ckpts/fusion_ckpts/blip2")
        self.tokenizer = T5Tokenizer.from_pretrained("../../ckpts/text_ckpts/molt5-base",  model_max_length=512)
        self.processor = Blip2Processor(self.processor.current_processor, self.tokenizer)
  
        
        self.model.ln_text = LayerNorm(768)
        self.model.vision_model = VisonEncoder()
        self.model.ln_vision = LayerNorm(self.model.vision_model.hidden_size)
        self.model.graph_encoder = GraphEncoder(config)
        self.model.ln_graph = LayerNorm(self.model.graph_encoder.hidden_size)
        

        gitformer = GITFormer(384,768)
        self.model.git_former = gitformer.Qformer
        self.model.query_tokens = gitformer.query_tokens
    
        embed_dim = 256
        self.device = device

        
        self.task_list = task
        self.inputs_modal = []
        self.outputs_modal = []

        for task in self.task_list:
            inputs_modal = task['inputs_modal']
            self.inputs_modal = self.inputs_modal+inputs_modal
            outputs_modal = task['outputs_modal']
            self.outputs_modal = self.outputs_modal+outputs_modal

            
        self.inputs_modal = list(set(self.inputs_modal))  
        self.outputs_modal = list(set(self.outputs_modal)) 
        
            
    def get_git_former_outputs(self, mol, inputs_modal, outputs_modal):
        language_model_inputs_image = None
        language_model_inputs_text = None
        language_model_inputs_graph = None
        input_tensors = []
        batch_size = len(mol['smiles'])

        prompt = generate_prompt(inputs_modal, outputs_modal)
        #print(f"prompt : {prompt}")
        prompt = self.processor(text=prompt, return_tensors="pt")
        input_ids = prompt['input_ids']
        attention_mask = prompt['attention_mask']
        mol['input_ids'] = input_ids.repeat(batch_size, 1).to(self.device)
        mol['attention_mask'] = attention_mask.repeat(batch_size, 1).to(self.device)
        if 'image2d' in inputs_modal:
            language_model_inputs_image = self.get_image_git_former_features(mol)
            input_tensors.append(language_model_inputs_image)
            
        if 'SMILES' in inputs_modal:
            language_model_inputs_text = self.get_text_git_former_features(mol, inputs_modal)
            input_tensors.append(language_model_inputs_text)
            
        if 'isoSMILES' in inputs_modal:
            language_model_inputs_text = self.get_text_git_former_features(mol, inputs_modal)
            input_tensors.append(language_model_inputs_text)
            
        if 'caption' in inputs_modal:
            language_model_inputs_text = self.get_text_git_former_features(mol, inputs_modal)
            input_tensors.append(language_model_inputs_text)

        if 'graph2d' in inputs_modal:
            language_model_inputs_graph = self.get_graph_git_former_features(mol)
            input_tensors.append(language_model_inputs_graph)

        # Stack tensors along a new dimension and compute the mean
        qformer_outputs = torch.stack(input_tensors).mean(dim=0)
        return qformer_outputs, mol
    
    def forward(self, mol):
        loss_list = []
        for i, task in enumerate(self.task_list):
            inputs_modal = task['inputs_modal']
            outputs_modal = task['outputs_modal']
            qformer_outputs, mol = self.get_git_former_outputs(mol, inputs_modal, outputs_modal)

            if('SMILES' in outputs_modal or 'caption' in outputs_modal or 'isoSMILES' in outputs_modal):
                loss_text = self.loss_text(mol, qformer_outputs, outputs_modal)
                loss_list.append(loss_text)

        loss_tensor = torch.stack(loss_list) 
        loss = torch.mean(loss_tensor)
        #loss = torch.sum(torch.stack(loss_list) * self.loss_weights_base_task())
        return loss

    def loss_text(self, mol, qformer_outputs, outputs_modal):
    
        language_model_inputs = qformer_outputs
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)
        
        inputs_embeds = self.model.language_model.get_input_embeddings()(mol['input_ids'])

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        attention_mask = mol['attention_mask']
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
        
        if('SMILES' in outputs_modal):
            labels = mol['smiles_labels']
        elif('isoSMILES' in outputs_modal):
            labels = mol['isosmiles_labels']
        elif('caption' in outputs_modal):
            labels = mol['caption_labels']
        """
        # use the encoder of MolT5
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            #inputs_embeds = language_model_inputs,
            attention_mask=attention_mask,
            #attention_mask = language_model_attention_mask,
            labels=labels
        )
        
        """
        
        # only use the decoder of MolT5
        h = BaseModelOutput(
            last_hidden_state=language_model_inputs,
            hidden_states=None,
            attentions=None
        )
        outputs = self.model.language_model(
            encoder_outputs = h,
            labels=labels
        )
   
        loss_text = outputs['loss']
        return loss_text
    
    def get_graph_git_former_features(self, mol):
        graph_embeds = self.model.ln_graph(self.model.graph_encoder(mol))
        graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(
            graph_embeds.device
        )
        query_tokens = self.model.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        #print(f"graph_query_tokens:{query_tokens.shape}")    
        #query_tokens:torch.Size([4, 32, 768])
        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_atts,
            modal='graph',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state
        #print(f"graph_query_output.shape :{query_output.shape}")
        language_model_inputs_graph = query_output
        return language_model_inputs_graph
    
    def get_text_git_former_features(self, mol, inputs_modal):
        if 'isoSMILES' in inputs_modal:
            text = mol['isoSMILES']
        if 'SMILES' in inputs_modal:
            text = mol['SMILES']
        if 'caption' in inputs_modal:
            text = mol['Caption']
        text_embeds = self.model.ln_text(self.model.git_former.bert(
                text['input_ids'],
                attention_mask=text['attention_mask'],
                return_dict=True,
            ).last_hidden_state)
        text_attention_mask = torch.ones(text_embeds.size()[:-1], dtype=torch.long, device=text_embeds.device)
        query_tokens = self.model.query_tokens.expand(text_embeds.shape[0], -1, -1)
         
        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=text_attention_mask,
            modal='cs_text',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs_text = query_output
        return language_model_inputs_text
    
    def get_image_git_former_features(self, mol):
        image_embeds = self.model.ln_vision(self.model.vision_model(mol))
    
        image_embeds = image_embeds.float()
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )
        image_embeds = self.model.vision_model(mol) 

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            modal='image',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs_image = query_output

        return language_model_inputs_image

    def generate_text(self, mol, inputs_modal, outputs_modal):
        generated_ids = self.generate_language(mol, inputs_modal, outputs_modal)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts
        
    def generate_language(self, mol, inputs_modal, outputs_modal):
        language_model_inputs, mol = self.get_git_former_outputs(mol, inputs_modal, outputs_modal)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        
        input_ids = mol['input_ids'] 
        attention_mask = mol['attention_mask']

        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        
        
        """
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            #inputs_embeds = language_model_inputs,
            attention_mask=attention_mask,
            #attention_mask = language_attention_mask,
            num_beams = 5,
            max_length = 512
        )
        """
        h = BaseModelOutput(
            last_hidden_state=language_model_inputs,
            hidden_states=None,
            attentions=None
        )
        outputs = self.model.language_model.generate(
            encoder_outputs = h,
            num_beams = 5,
            max_length = 512
        )
        return outputs
    
class VisonEncoder(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        #encoder
        self.image_encoder = SwinTransformer()
        image_ckpt = "../../ckpts/image_ckpts/swin_transform_focalloss.pth"
        image_ckpt = torch.load(image_ckpt, map_location='cpu')
        self.image_encoder.load_state_dict(image_ckpt['encoder'], strict=False)
        self.num_features = 1536
        self.hidden_size = 768

        self.fc_hidden = nn.Linear(self.num_features, self.hidden_size)
        
        
    def forward(self, mol):
        image2ds = mol['image2d']
        image2d_embeddings = self.image_encoder(image2ds)
        image2d_embeddings = self.fc_hidden(image2d_embeddings)
        return image2d_embeddings
    
class GraphEncoder(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        self.graph2d_encoder = MoMu(config["graph"]).graph_encoder
    
        for param in self.graph2d_encoder.parameters():
            param.requires_grad = False
        
        self.num_features = 300
        self.hidden_size = 768
        self.fc_hidden = nn.Linear(self.num_features, self.hidden_size)
    
    def forward(self, mol):
        graph_feats, node_feats, node_feats_mask = self.graph2d_encoder(mol["graph2d"])
        node_feats = self.fc_hidden(node_feats)
        return node_feats
 

