import json
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
import numpy as np
import random

def bert_featurize(txt,bert_model,bert_tokenizer):
    tokenized_text = bert_tokenizer.tokenize(txt)
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    n = len(tokenized_text)
    segments_ids = [0 for _ in range(n)]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
    last_layer_tensor = encoded_layers[-1].squeeze()
    return last_layer_tensor

def get_amazon_bert_features_labels(n_samples=300,AMAZON_REVIEWS_PATH='data/reviews_Toys_and_Games_5.json',max_review_char_length=500):
    bert_pretrained_version = 'bert-base-cased'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained_version)
    bert_model = BertModel.from_pretrained(bert_pretrained_version)
    bert_model.eval()

    with open(AMAZON_REVIEWS_PATH,'r') as f:
        df = pd.DataFrame([json.loads(line) for line in f])
    df_sampled = df.sample(n_samples,random_state=0)
    X = np.vstack([bert_featurize(txt[:max_review_char_length],bert_model,bert_tokenizer)[0,:].detach().numpy() for txt in list(df_sampled['reviewText'])])
    Y = np.array(df_sampled['overall']>4.0).astype(np.int)
    return X,Y
