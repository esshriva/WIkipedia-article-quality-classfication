import re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tqdm.notebook import tqdm
import mwparserfromhell
from tqdm import tqdm
import nltk
import numpy as np
import pandas as pd
pd.set_option('chained_assignment', None)

def apply_on_training_set():
    train_dataset = pd.read_csv('./dataset/train_dataset_basic_ft.csv', usecols=['text', 'labels'])

    train_dataset['pos_counts'] = ''

    tokenizer = RegexpTokenizer(r'\w+')


    for idx, row in tqdm(train_dataset.iterrows()):
        text = row['text']
        wikicode = mwparserfromhell.parse(text)
        clean_text = ' '.join(list(map(str, wikicode.filter_text())))
        # tokens = tokenizer.tokenize(text)

        tokens = tokenizer.tokenize(clean_text)

        tokens_without_sw = [word for word in tokens if not word in stopwords.words('english')]

        pos_tags = nltk.pos_tag(tokens_without_sw)
        counts = dict(Counter(np.matrix(pos_tags)[:, 1].flatten().tolist()[0]))

        train_dataset['pos_counts'][idx] = counts

    train_dataset['only_pos'] = train_dataset['pos_counts'].apply(lambda x: list(x.keys()))

    all_pos_list = []
    for i in tqdm(range(train_dataset.shape[0])):
        all_pos_list.extend(train_dataset['only_pos'][i])

    all_pos_set = set(all_pos_list)
    print(len(all_pos_set))

    matrix = np.zeros(shape=(train_dataset.shape[0], len(all_pos_set)+1))
    pos_feature_df = pd.DataFrame(data=matrix, columns=['sent_id'] + list(all_pos_set))

    pos_feature_df['sent_id'] = train_dataset['text'].copy()

    for i in tqdm(range(train_dataset.shape[0])):
        pos_dict = train_dataset['pos_counts'][i]
        for key, value in pos_dict.items():
            pos_feature_df[key][i] = value
    pos_feature_df.to_csv('./dataset/train_dataset_pos_feature.csv', index=False)

def apply_on_testing_set():
    test_dataset = pd.read_csv('./dataset/test_dataset_basic_ft.csv', usecols=['text', 'label'])

    test_dataset['pos_counts'] = ''

    tokenizer = RegexpTokenizer(r'\w+')

    for idx, row in tqdm(test_dataset.iterrows()):
        text = row['text']
        wikicode = mwparserfromhell.parse(text)
        clean_text = ' '.join(list(map(str, wikicode.filter_text())))
        # tokens = tokenizer.tokenize(text)

        tokens = tokenizer.tokenize(clean_text)

        tokens_without_sw = [word for word in tokens if not word in stopwords.words('english')]

        pos_tags = nltk.pos_tag(tokens_without_sw)
        counts = dict(Counter(np.matrix(pos_tags)[:, 1].flatten().tolist()[0]))

        test_dataset['pos_counts'][idx] = counts

    test_dataset['only_pos'] = test_dataset['pos_counts'].apply(lambda x: list(x.keys()))

    all_pos_list = []
    for i in tqdm(range(test_dataset.shape[0])):
        all_pos_list.extend(test_dataset['only_pos'][i])

    all_pos_set2 = set(all_pos_list)
    print(len(all_pos_set2))

    matrix = np.zeros(shape=(test_dataset.shape[0], len(all_pos_set2)+1))
    pos_feature_df = pd.DataFrame(data=matrix, columns=['sent_id'] + list(all_pos_set2))

    pos_feature_df['sent_id'] = test_dataset['text'].copy()

    for i in tqdm(range(test_dataset.shape[0])):
        pos_dict = test_dataset['pos_counts'][i]
        for key, value in pos_dict.items():
            pos_feature_df[key][i] = value
    pos_feature_df.to_csv('./dataset/test_dataset_pos_feature.csv', index=False)

if __name__ == '__main__':
    # apply_on_training_set()
    # apply_on_testing_set()
    pass