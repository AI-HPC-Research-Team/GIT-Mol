# -*- coding: utf-8 -*-

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig
from peft import get_peft_model

from transformers import BertTokenizer
from models.drug_encoder.swin_transformer import SwinTransformer

from models.drug_encoder.momu import MoMu
from models.blip2.GITformer import BertConfig, BertLMHeadModel

import rdkit.Chem as Chem
import numpy as np

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class GITFormer(nn.Module):
    def __init__(self, num_query_token, vision_graph_width, cross_attention_freq = 2):
        super().__init__()
        encoder_config = BertConfig.from_pretrained("../../ckpts/text_ckpts/scibert_scivocab_uncased")
        encoder_config.encoder_width = vision_graph_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        self.Qformer = BertLMHeadModel.from_pretrained(
            "../../ckpts/text_ckpts/scibert_scivocab_uncased", config=encoder_config
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)


class GITModel(nn.Module):
    def __init__(self, config = None, fp = False, modal = None, device = None):
        super().__init__()

        self.blip2conf = Blip2Config()
        self.model = Blip2ForConditionalGeneration(self.blip2conf)



        self.tokenizer = BertTokenizer.from_pretrained("../../ckpts/text_ckpts/scibert_scivocab_uncased", truncation_side='right')
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        self.model.ln_text = LayerNorm(768)
        self.model.vision_model = VisonEncoder()
        self.model.ln_vision = LayerNorm(self.model.vision_model.hidden_size)
        self.model.graph_encoder = GraphEncoder(config)
        self.model.ln_graph = LayerNorm(self.model.graph_encoder.hidden_size)

        """
        language_model = T5ForConditionalGeneration.from_pretrained("../../ckpts/text_ckpts/molt5-large")
        language_lora_config = LoraConfig(
            peft_type="LORA",
            r=16,  # rank of the update matrices
            lora_alpha=16,  # scaling factor for LoRA updates
            target_modules=["q","v","lm_head","shared"],  # apply LoRA to query and value matrices
            lora_dropout=0.1,  # dropout rate for LoRA updates
            bias="none",  # do not train bias parameters
                #modules_to_save=["lm_head"]  # also train the classifier parameters
        )
        self.model.language_model  = get_peft_model(language_model, language_lora_config)
        """
        gitformer = GITFormer(384,768)
        self.model.git_former = gitformer.Qformer
        self.model.query_tokens = gitformer.query_tokens
        
        self.itm_head = nn.Linear(self.model.git_former.config.hidden_size, 2)
        self.gtm_head = nn.Linear(self.model.git_former.config.hidden_size, 2)
        self.ctm_head = nn.Linear(self.model.git_former.config.hidden_size, 2)

        embed_dim = 256
        self.vision_proj = nn.Linear(self.model.git_former.config.hidden_size, embed_dim)
        self.graph_proj = nn.Linear(self.model.git_former.config.hidden_size, embed_dim)
        self.cs_text_proj = nn.Linear(self.model.git_former.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.model.git_former.config.hidden_size, embed_dim)
        self.model_freeze()
        self.device = device
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        #self.task = ['itm','itc','gtm','gtc']
        self.task = []
        self.modal = modal
        self.input_modal = modal['inputs_modal']
        self.output_modal = modal['outputs_modal']
        if('isoSMILES' in self.output_modal):
            if('image2d' in self.input_modal):
                self.task.append('itm')
                self.task.append('itc')
            if('caption' in self.input_modal):
                self.task.append('ctm')
                self.task.append('ctc')
        if('caption' in self.output_modal):
            if('image2d' in self.input_modal):
                self.task.append('itm')
                self.task.append('itc')
            if('isoSMILES' in self.input_modal):
                self.task.append('ctm')
                self.task.append('ctc')
            if('graph2d' in self.input_modal):
                self.task.append('gtm')
                self.task.append('gtc')
                
    def model_freeze(self):
        
        for param in self.model.graph_encoder.parameters():
            param.requires_grad = False
           
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
    
    def forward(self, mol):
        #image = mol["image2d"]
        loss = 0
        text_modal = self.output_modal[0]
        if(text_modal=='caption'):
            text = mol['Caption']
        else:
            text = mol[text_modal]
        batch_size = text['input_ids'].size(0)
        if('image2d' in self.input_modal):
            image_embeds = self.model.ln_vision(self.model.vision_model(mol))
            #print(f"image_embeds : {image_embeds.shape}")
            image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )
            image_targets = torch.arange(batch_size).to(image_embeds.device)
        if('graph2d' in self.input_modal):
            graph_embeds = self.model.ln_graph(self.model.graph_encoder(mol))
            graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(
                graph_embeds.device
            )
            graph_targets = torch.arange(batch_size).to(graph_embeds.device)
            
        if('caption' in self.input_modal):
            cs_text = mol['Caption']
            cs_text_embeds = self.model.git_former.bert(
                cs_text['input_ids'],
                attention_mask=cs_text['attention_mask'],
                return_dict=True,
            ).last_hidden_state
            cs_text_atts = torch.ones(cs_text_embeds.size()[:-1], dtype=torch.long).to(
                cs_text_embeds.device
            )
            cs_text_targets = torch.arange(batch_size).to(cs_text_embeds.device) 
            
        if('isoSMILES' in self.input_modal):
            cs_text = mol['isoSMILES']
            cs_text_embeds = self.model.git_former.bert(
                cs_text['input_ids'],
                attention_mask=cs_text['attention_mask'],
                return_dict=True,
            ).last_hidden_state
            cs_text_atts = torch.ones(cs_text_embeds.size()[:-1], dtype=torch.long).to(
                cs_text_embeds.device
            )
            cs_text_targets = torch.arange(batch_size).to(cs_text_embeds.device) 
            
        text_output = self.model.git_former.bert(
            text['input_ids'],
            attention_mask=text['attention_mask'],
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        if("itm" in self.task):
            
            # Initializing lists to hold the original and negative samples
            image_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(image_embeds.shape[0]):
                # Original samples
                image_embeds_list.append(image_embeds[i])
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

                # Negative samples (neg_text_input_ids corresponds to image_embeds)
                neg_text_input_ids = text['input_ids'][i-1] if i == image_embeds.shape[0] - 1 else text['input_ids'][i+1]
                neg_text_attention_mask = text['attention_mask'][i-1] if i == image_embeds.shape[0] - 1 else text['attention_mask'][i+1]
                text_input_ids_list.append(neg_text_input_ids)
                text_attention_mask_list.append(neg_text_attention_mask)
                image_embeds_list.append(image_embeds[i])

                # Negative samples (text_input_ids corresponds to neg_image_embeds)
                neg_image_embeds = image_embeds[i-1] if i == image_embeds.shape[0] - 1 else image_embeds[i+1]
                image_embeds_list.append(neg_image_embeds)
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

            # Stack all samples into two large tensors
            image_embeds_all = torch.stack(image_embeds_list, dim=1).reshape(-1, image_embeds.size(1), image_embeds.size(2))
            text_input_ids_all = torch.stack(text_input_ids_list, dim=1).reshape(-1, text['input_ids'].size(1))
            text_attenetion_mask_all = torch.stack(text_attention_mask_list, dim=1).reshape(-1, text['attention_mask'].size(1))
            # Create image attention masks for the concatenated tensor
            image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
                image_embeds_all.device
            )
            query_tokens_itm = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
                image_embeds_all.device
            )
            attention_mask_all = torch.cat([query_atts_itm, text_attenetion_mask_all], dim=1)
            
            output_itm = self.model.git_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                modal='image',
                return_dict=True,
            )
            itm_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
            
            itm_logit = self.itm_head(itm_embeddings)
            itm_logit = itm_logit.mean(dim=1)
            #itm_logit = self.itm_head(itm_embeddings)
            # Create labels: 1 for the original samples, 0 for the negative samples
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 2)], dim=0).long().to(itm_logit.device)

            # Calculate cross entropy loss
            loss_itm = F.cross_entropy(itm_logit, labels)
        
            loss = loss+loss_itm
        
        if("gtm" in self.task):
            
            # Initializing lists to hold the original and negative samples
            graph_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(graph_embeds.shape[0]):
                # Original samples
                graph_embeds_list.append(graph_embeds[i])
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

                # Negative samples (neg_text_input_ids corresponds to image_embeds)
                neg_text_input_ids = text['input_ids'][i-1] if i == graph_embeds.shape[0] - 1 else text['input_ids'][i+1]
                neg_text_attention_mask = text['attention_mask'][i-1] if i == graph_embeds.shape[0] - 1 else text['attention_mask'][i+1]
                text_input_ids_list.append(neg_text_input_ids)
                text_attention_mask_list.append(neg_text_attention_mask)
                graph_embeds_list.append(graph_embeds[i])

                # Negative samples (text_input_ids corresponds to neg_image_embeds)
                neg_graph_embeds = graph_embeds[i-1] if i == graph_embeds.shape[0] - 1 else graph_embeds[i+1]
                graph_embeds_list.append(neg_graph_embeds)
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

            # Stack all samples into two large tensors
            graph_embeds_all = torch.stack(graph_embeds_list, dim=1).reshape(-1, graph_embeds.size(1), graph_embeds.size(2))
            text_input_ids_all = torch.stack(text_input_ids_list, dim=1).reshape(-1, text['input_ids'].size(1))
            text_attenetion_mask_all = torch.stack(text_attention_mask_list, dim=1).reshape(-1, text['attention_mask'].size(1))
            # Create image attention masks for the concatenated tensor
            graph_atts_all = torch.ones(graph_embeds_all.size()[:-1], dtype=torch.long).to(
                graph_embeds_all.device
            )
            query_tokens_gtm = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_atts_gtm = torch.ones(query_tokens_gtm.size()[:-1], dtype=torch.long).to(
                graph_embeds_all.device
            )
            attention_mask_all = torch.cat([query_atts_gtm, text_attenetion_mask_all], dim=1)
            
            output_gtm = self.model.git_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_gtm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                modal = 'graph',
                return_dict=True,
            )
            gtm_embeddings = output_gtm.last_hidden_state[:, : query_tokens_gtm.size(1), :]
            
            gtm_logit = self.gtm_head(gtm_embeddings)
            gtm_logit = gtm_logit.mean(dim=1)
            #itm_logit = self.itm_head(itm_embeddings)
            # Create labels: 1 for the original samples, 0 for the negative samples
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 2)], dim=0).long().to(gtm_logit.device)

            # Calculate cross entropy loss
            loss_gtm = F.cross_entropy(gtm_logit, labels)

            loss = loss+loss_gtm
            
        if("ctm" in self.task):
            # Initializing lists to hold the original and negative samples
            cs_text_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(cs_text_embeds.shape[0]):
                # Original samples
                cs_text_embeds_list.append(cs_text_embeds[i])
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

                # Negative samples (neg_text_input_ids corresponds to image_embeds)
                neg_text_input_ids = text['input_ids'][i-1] if i == cs_text_embeds.shape[0] - 1 else text['input_ids'][i+1]
                neg_text_attention_mask = text['attention_mask'][i-1] if i == cs_text_embeds.shape[0] - 1 else text['attention_mask'][i+1]
                text_input_ids_list.append(neg_text_input_ids)
                text_attention_mask_list.append(neg_text_attention_mask)
                cs_text_embeds_list.append(cs_text_embeds[i])

                # Negative samples (text_input_ids corresponds to neg_image_embeds)
                neg_cs_text_embeds = cs_text_embeds[i-1] if i == cs_text_embeds.shape[0] - 1 else cs_text_embeds[i+1]
                cs_text_embeds_list.append(neg_cs_text_embeds)
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

            # Stack all samples into two large tensors
            cs_text_embeds_all = torch.stack(cs_text_embeds_list, dim=1).reshape(-1, cs_text_embeds.size(1), cs_text_embeds.size(2))
            text_input_ids_all = torch.stack(text_input_ids_list, dim=1).reshape(-1, text['input_ids'].size(1))
            text_attenetion_mask_all = torch.stack(text_attention_mask_list, dim=1).reshape(-1, text['attention_mask'].size(1))
            # Create image attention masks for the concatenated tensor
            cs_text_atts_all = torch.ones(cs_text_embeds_all.size()[:-1], dtype=torch.long).to(
                cs_text_embeds_all.device
            )
            query_tokens_ctm = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_atts_ctm = torch.ones(query_tokens_ctm.size()[:-1], dtype=torch.long).to(
                cs_text_embeds_all.device
            )
            attention_mask_all = torch.cat([query_atts_ctm, text_attenetion_mask_all], dim=1)
            
            output_ctm = self.model.git_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_ctm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=cs_text_embeds_all,
                encoder_attention_mask=cs_text_atts_all,
                modal = 'cs_text',
                return_dict=True,
            )
            ctm_embeddings = output_ctm.last_hidden_state[:, : query_tokens_ctm.size(1), :]
            
            ctm_logit = self.ctm_head(ctm_embeddings)
            ctm_logit = ctm_logit.mean(dim=1)
            #itm_logit = self.itm_head(itm_embeddings)
            # Create labels: 1 for the original samples, 0 for the negative samples
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 2)], dim=0).long().to(ctm_logit.device)

            # Calculate cross entropy loss
            loss_ctm = F.cross_entropy(ctm_logit, labels)

            loss = loss+loss_ctm
            
        if("itc" in self.task):
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.model.git_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                modal = 'image',
                return_dict=True,
            )
            image_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )

            sim_q2t = torch.matmul(
                image_feats.unsqueeze(1), text_feat.unsqueeze(-1)
            ).squeeze()
                # [batch_size, batch_size*num_gpu, num_query_tokens]

                # image-text similarity: aggregate across all query tokens
            sim_i2t, _ = sim_q2t.max(-1)
            sim_i2t = sim_i2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)
            ).squeeze()

                # text-image similarity: aggregate across all query tokens
            sim_t2i, _ = sim_t2q.max(-1)
            sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
            loss_itc = (
                F.cross_entropy(sim_i2t, image_targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, image_targets, label_smoothing=0.1)
                ) / 2

            loss = loss+loss_itc
        
        
        if("gtc" in self.task):
            query_tokens = self.model.query_tokens.expand(graph_embeds.shape[0], -1, -1)

            query_output = self.model.git_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_atts,
                modal = 'graph',
                return_dict=True,
            )
            
            graph_feats = F.normalize(
                self.graph_proj(query_output.last_hidden_state), dim=-1
            )

            sim_q2t = torch.matmul(
                graph_feats.unsqueeze(1), text_feat.unsqueeze(-1)
            ).squeeze()
                # [batch_size, batch_size*num_gpu, num_query_tokens]

                # image-text similarity: aggregate across all query tokens
            sim_g2t, _ = sim_q2t.max(-1)
            sim_g2t = sim_g2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), graph_feats.permute(0, 2, 1)
            ).squeeze()

                # text-image similarity: aggregate across all query tokens
            sim_t2g, _ = sim_t2q.max(-1)
            sim_t2g = sim_t2g / self.temp  # [batch_size, batch_size*num_gpu]
            loss_gtc = (
                F.cross_entropy(sim_g2t, graph_targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2g, graph_targets, label_smoothing=0.1)
                ) / 2

            loss = loss+loss_gtc
            
        if("ctc" in self.task):
            query_tokens = self.model.query_tokens.expand(cs_text_embeds.shape[0], -1, -1)

            query_output = self.model.git_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=cs_text_embeds,
                encoder_attention_mask=cs_text_atts,
                modal = 'cs_text',
                return_dict=True,
            )
            
            cs_text_feats = F.normalize(
                self.cs_text_proj(query_output.last_hidden_state), dim=-1
            )

            sim_q2t = torch.matmul(
                cs_text_feats.unsqueeze(1), text_feat.unsqueeze(-1)
            ).squeeze()
                # [batch_size, batch_size*num_gpu, num_query_tokens]

                # image-text similarity: aggregate across all query tokens
            sim_c2t, _ = sim_q2t.max(-1)
            sim_c2t = sim_c2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), cs_text_feats.permute(0, 2, 1)
            ).squeeze()

                # text-image similarity: aggregate across all query tokens
            sim_t2c, _ = sim_t2q.max(-1)
            sim_t2c = sim_t2c / self.temp  # [batch_size, batch_size*num_gpu]
            loss_ctc = (
                F.cross_entropy(sim_c2t, cs_text_targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2c, cs_text_targets, label_smoothing=0.1)
                ) / 2

            loss = loss+loss_ctc
            
        loss = loss/len(self.task)
        return loss


class VisonEncoder(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        #encoder
        self.image_encoder = SwinTransformer()
        image_ckpt = "ckpts/image_ckpts/swin_transform_focalloss.pth"
        image_ckpt = torch.load(image_ckpt, map_location='cpu')
        self.image_encoder.load_state_dict(image_ckpt['encoder'], strict=False)
        self.num_features = 1536
        self.hidden_size = 768
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        self.fc_hidden = nn.Linear(self.num_features, self.hidden_size)
        
        
    def forward(self, mol):
        image2ds = mol['image2d']
        image2d_embeddings = self.image_encoder(image2ds)
        image2d_embeddings = self.fc_hidden(image2d_embeddings)
        return image2d_embeddings

class GraphEncoder(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


