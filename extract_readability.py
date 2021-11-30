import pandas as pd
import numpy as np
import mwparserfromhell
import textstat
from tqdm.notebook import tqdm
import re
pd.set_option('chained_assignment', None)

def apply_on_training_set():
    train_dataset = pd.read_csv("./dataset/train_dataset_basic_ft.csv", usecols=['text', 'labels'])
    dirty_text = train_dataset['text'].apply(lambda text: re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)).tolist()

    new_df = pd.DataFrame(columns=['id', 'flesh_reading', 'flesh_kincaid', 'smog_index', 'coleman_liau', 'automated_readabilily', 'difficult_words', 'dale_chall_readability', 'linsear_write_formula', 'gunning_fog'])

    for i in tqdm(range(len(dirty_text))):
        feat = []
        wikicode = mwparserfromhell.parse(dirty_text[i])
        clean_text = ' '.join(list(map(str, wikicode.filter_text())))

        feat.append(i)
        feat.append(textstat.flesch_reading_ease(clean_text))
        feat.append(textstat.flesch_kincaid_grade(clean_text))
        feat.append(textstat.smog_index(clean_text))
        feat.append(textstat.coleman_liau_index(clean_text))
        feat.append(textstat.automated_readability_index(clean_text))
        feat.append(textstat.difficult_words(clean_text))
        feat.append(textstat.dale_chall_readability_score_v2(clean_text))
        feat.append(textstat.linsear_write_formula(clean_text))
        feat.append(textstat.gunning_fog(clean_text))

        new_df.loc[new_df.shape[0]] = feat

    new_df['labels'] = train_dataset['labels']
    new_df.to_csv('./dataset/train_dataset_readability.csv', index=False)

def apply_on_testing_set():
    test_dataset = pd.read_csv("./dataset/test_dataset_basic_ft.csv", usecols=['text', 'label'])
    dirty_text = test_dataset['text'].apply(lambda text: re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)).tolist()

    new_df = pd.DataFrame(columns=['id', 'flesh_reading', 'flesh_kincaid', 'smog_index', 'coleman_liau', 'automated_readabilily', 'difficult_words', 'dale_chall_readability', 'linsear_write_formula', 'gunning_fog'])

    for i in tqdm(range(len(dirty_text))):
        feat = []
        wikicode = mwparserfromhell.parse(dirty_text[i])
        clean_text = ' '.join(list(map(str, wikicode.filter_text())))

        feat.append(i)
        feat.append(textstat.flesch_reading_ease(clean_text))
        feat.append(textstat.flesch_kincaid_grade(clean_text))
        feat.append(textstat.smog_index(clean_text))
        feat.append(textstat.coleman_liau_index(clean_text))
        feat.append(textstat.automated_readability_index(clean_text))
        feat.append(textstat.difficult_words(clean_text))
        feat.append(textstat.dale_chall_readability_score_v2(clean_text))
        feat.append(textstat.linsear_write_formula(clean_text))
        feat.append(textstat.gunning_fog(clean_text))

        new_df.loc[new_df.shape[0]] = feat

    new_df['labels'] = test_dataset['label']
    new_df.to_csv('./dataset/test_dataset_readability.csv', index=False)

if __name__ == '__main__':
    apply_on_training_set()
    apply_on_testing_set()