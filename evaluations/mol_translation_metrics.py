'''
Code from https://github.com/blender-nlp/MolT5
```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''


import pickle
import argparse
import csv

import os.path as osp

import numpy as np


from nltk.translate.bleu_score import corpus_bleu

from Levenshtein import distance as lev
import pandas as pd
from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def mol_evaluate(targets, preds, descriptions, verbose=False):
    outputs = []

    for i in range(len(targets)):
            gt_smi = targets[i]
            ot_smi = preds[i]
            outputs.append((descriptions[i], gt_smi, ot_smi))


    bleu_scores = []
    

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):

        if i % 100 == 0:
            if verbose:
                print(i, 'processed.')


        gt_tokens = [c for c in gt]

        out_tokens = [c for c in out]

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        # mscore = meteor_score([gt], out)
        # meteor_scores.append(mscore)

    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    if verbose: print('BLEU score:', bleu_score)

    # Meteor score
    # _meteor_score = np.mean(meteor_scores)
    # print('Average Meteor score:', _meteor_score)

    rouge_scores = []

    references = []
    hypotheses = []

    levs = []

    num_exact = 0

    bad_mols = 0

    result_dataframe = pd.DataFrame(outputs, columns=['summary', 'ground truth', 'isosmiles'])
    result_dataframe['exact'] =0
    result_dataframe['valid'] =0
    result_dataframe['lev'] =1000
    #print(result_dataframe.head())
    for i, (smi, gt, out) in enumerate(outputs):

        hypotheses.append(out)
        references.append(gt)

        try:
            m_out = Chem.MolFromSmiles(out)
            m_gt = Chem.MolFromSmiles(gt)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): 
                num_exact += 1
                result_dataframe.at[i, 'exact'] = 1
            else:
                result_dataframe.at[i, 'exact'] = 0
            #if gt == out: num_exact += 1 #old version that didn't standardize strings
            result_dataframe.at[i, 'valid'] = 1
        except:
            bad_mols += 1
            result_dataframe.at[i, 'valid'] = 0

        

        levs.append(lev(out, gt))
        #result_dataframe.iloc[i]['lev'] = lev(out, gt)
        result_dataframe.at[i, 'lev'] = lev(out, gt)

    # Exact matching score
    exact_match_score = num_exact/(i+1)
    if verbose:
        print('Exact Match:')
        print(exact_match_score)

    # Levenshtein score
    levenshtein_score = np.mean(levs)
    if verbose:
        print('Levenshtein:')
        print(levenshtein_score)
        
    validity_score = 1 - bad_mols/len(outputs)
    if verbose:
        print('validity:', validity_score)

    return bleu_score, exact_match_score, levenshtein_score, validity_score, result_dataframe
