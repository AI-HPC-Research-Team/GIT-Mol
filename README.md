##  GIT-Mol
Here, we introduce GIT-Mol, a multi-modal large language model that integrates the structure Graph, Image, and Text information, including the Simplified Molecular Input Line Entry System (SMILES) and molecule captions. To facilitate the integration of multi-modal molecular data, we propose GIT-Former, a novel paradigm capable of mapping all modalities into a unified latent space.


### Data
#### pretrain_data
igdata                       # This folder contains the data for pretraining GIT-Former with image, graph, and SMILES modalities.
├── train_4m.pkl										 
└── valid_400k.pkl
igcdata                      # This folder contains the data for pretraining GIT-Former with image, graph, and caption modalities.
├── train_220k.pkl										 
└── valid_20k.pkl
image2d                      # Data of molecule images in the pretrain stage
└──cid.png

#### finetune_data             
ChEBI-20                     # This folder contains the data for finetuning GIT-Mol on molecule generation(caption->SMILES)   
├──train_26k.txt
├──validation_3k.txt
└──test_3k.txt
molcap                       # This folder contains the data for finetuning GIT-Mol on molecule caption(any->caption) and molecule image caption(image->SMILES, image->caption)
├──train_72k.pkl
├──valid_9k.pkl
├──test_9k.pkl
└──image2d
    └──cid.png
MoleculeNet                  # This folder contains the data for finetuning GIT-Mol for molecule properties prediction (classification)
├──bbbp
├──bace
├──tox21
├──clintox
├──sider
└──toxcast

Due to file size constraints, the ChEBI-20 and MoleculeNet datasets can be downloaded from the following link:
https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data
https://moleculenet.org/datasets-1


#### data processing
data_processing.ipynb


### Model
GIT-MOL
├──ckpts                                # This folder contains checkpoints of pretraining and finetuning
├──configs
|   ├──config.json                      # Config file of this model
|   └──deepspeed_config.json            # Config file of deepspeed in Accelerate
├──models
|   ├──GIT_Former.py                    # Code of GIT-Former
|   ├──momu.py                          # Code of the graph encoder
|   ├──momu_gnn.py                      # Code of the graph encoder
|   ├──swin_transformer.py              # Code of the image encoder
|   ├──model_pretrain.py                # Code of the pretraining model
|   └──model_finetune.py                # Code of the finetuning model 
├──dataset
|   ├──dataset.py                       
|   └──graph_featurizer.py
└──utils
    └──utils.py


### Training and Evaluation
GIT-MOL
├──evaluations                          # Evaluations of molecule translation tasks
|   ├──fingerprint_metrics.py
|   ├──text_translation_metrics.py
|   └──mol_translation_metrics.py
└──train
    ├──pretrain.py                   
    └──finetune
        ├──molecule_translation.py      # Finetuning of molecule translation task
        └──property_prediction
                ├──finetune.py          # Finetuning of molecule properties prediction task
                ├──model.py             
                ├──splitters.py
                └──loader.py
				
#### property_prediction -- finetune.py 
--modals 
	Modalities used in this task, contain graph2d, SMILES, or both
--pool, type=str, default = 'avg'
	Pooling function of text and graph embeddings.(Avg or Max)
--fusion_mode, type=str, default = 'attention' 
	If we use graph2d and SMILES modalities in this task, we can choose the fusion mode of the two embeddings.(Attention or Weights)
