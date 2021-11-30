import multiprocessing
import torch
import torch.nn as nn

import transformers
import pandas as pd
import numpy as np
pd.set_option('chained_assignment', None)
from tqdm import tqdm
import pickle as pb

def get_embeddings(model, tokenizer, df, start_idx, end_idx, is_test=False):
    name = None
    if is_test:
        name = 'test'
    else:
        if start_idx == 0 and end_idx == 6620:
            print('Working on 1st set')
            name = 1
        elif start_idx == 6620 and end_idx == 13240:
            print('Working on 2nd set')
            name = 2
        elif start_idx == 13240 and end_idx == 19860:
            print('Working on 3rd set')
            name = 3
        elif start_idx == 19860:
            print('Working on 4th set')
            name = 4
        else:
            name = 0

    dataset = df.iloc[start_idx:end_idx]
    dataset.reset_index(inplace=True, drop=True)
    doc_embeddings = []
    indices = []
    fixed_window_size = 256
    with torch.no_grad():
        for i in tqdm(range(dataset.shape[0])):
            tokens = tokenizer(dataset.iloc[i]['text'], return_tensors='pt')
            intermediate_embeds = []
            # print(dataset.iloc[i]['text'][:10])
            if tokens.input_ids.shape[1] <= fixed_window_size:
                tokens = tokenizer(dataset.iloc[i]['text'], return_tensors='pt', max_length=fixed_window_size, padding='max_length')
                # print()
                intermediate_embeds.append(model(**tokens).last_hidden_state)
            else:
                # print('Token Length:', tokens.input_ids.shape[1])
                for j in range(0, tokens.input_ids.shape[1], fixed_window_size):
                    start = j
                    end = start + fixed_window_size
                    if end >= tokens.input_ids.shape[1]:
                        # print('End iteration: Start:', start)
                        inputs = torch.zeros(fixed_window_size, dtype=torch.int64)
                        attn_masks = torch.zeros(fixed_window_size, dtype=torch.int64)

                        shp = tokens.input_ids[0][start:].numpy().shape[0]

                        inputs[:shp] = tokens.input_ids[0][start:]
                        attn_masks[:shp] = tokens.attention_mask[0][start:]

                        inps = inputs.reshape(1, -1)
                        attn_mask = attn_masks.reshape(1, -1)
                        # print(inps.shape)
                        intermediate_embeds.append(model(inps, attention_mask=attn_mask).last_hidden_state)
                    else:
                        # print('Start, End:', start, end)
                        inps = tokens.input_ids[0][start:end].reshape(1, -1)
                        attn_mask = tokens.attention_mask[0][start:end].reshape(1, -1)
                        # print(inps.shape)
                        intermediate_embeds.append(model(inps, attention_mask=attn_mask).last_hidden_state)
            doc_embeddings.append(torch.mean(torch.mean(torch.vstack([t for t in intermediate_embeds]), 0), 0))
            
            # doc_embeddings.append(list(intermediate_embeds))
            # indices.append(i)
            # with open(f'./vectors/{name}/{i}_embeddings.pt', 'wb') as f:
            #     pb.dump({'vector':intermediate_embeds}, f)
        print('Finished.......')
        
        if not is_test:
            with open(f'doc_embedding_{start_idx}_{end_idx}.pt', 'wb') as f:
                content = {'doc_embed': doc_embeddings, 'index': indices}
                pb.dump(content, f)
        else:
            with open(f'test_doc_embedding_{start_idx}_{end_idx}.pt', 'wb') as f:
                content = {'doc_embed': doc_embeddings, 'index': indices}
                pb.dump(content, f)
        
        # for i in range(len(doc_embeddings)):
        #     print(doc_embeddings[i].shape)


if __name__ == '__main__':
    model_name = 'bert-base-cased'

    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    model = transformers.BertModel.from_pretrained(model_name)

    # Train
    # dataset = pd.read_csv('./dataset/train_dataset.csv', usecols=['text', 'labels'])
    
    # Test
    dataset = pd.read_csv('./dataset/test_dataset.csv', usecols=['text', 'label'])

    # Train Dataset
    # 1
    # multiprocessing.Process(target=get_embeddings, args=(model, tokenizer, dataset, 0, 6620)).start()

    # 2
    # multiprocessing.Process(target=get_embeddings, args=(model, tokenizer, dataset, 6620, 13240)).start()

    # 3
    # multiprocessing.Process(target=get_embeddings, args=(model, tokenizer, dataset, 13240, 19860)).start()

    # 4
    # multiprocessing.Process(target=get_embeddings, args=(model, tokenizer, dataset, 19860, dataset.shape[0])).start()

    # Test dataset
    multiprocessing.Process(target=get_embeddings, args=(model, tokenizer, dataset, 0, dataset.shape[0], True)).start()