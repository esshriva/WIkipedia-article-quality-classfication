import pandas as pd
import numpy as np
import textstat
from tqdm import tqdm
import re
pd.set_option('chained_assignment', None)

train_dataset = pd.read_csv("./dataset/training_set.csv")
test_dataset = pd.read_csv("./dataset/testing_set.csv")

# Trainin Set Structural Features
train_dataset['article_length_bytes'] = train_dataset['text'].apply(lambda s: len(s.encode('utf-8')))
train_dataset['no_of_citations'] = train_dataset['text'].apply(lambda s: s.count('<ref>'))
train_dataset['no_of_non_citations'] = train_dataset['text'].apply(lambda s: s.count('{{cn}}'))
train_dataset['no_of_infobox'] = train_dataset['text'].apply(lambda s: s.count('{{Infobox'))
train_dataset['no_of_categories'] = 0
train_dataset['no_external_links'] = 0
train_dataset['no_internal_links'] = 0
train_dataset['no_level_2_headers'] = 0
train_dataset['no_level_3_plus_headers'] = 0
train_dataset['no_of_images'] = 0
train_dataset['no_of_references'] = 0

for idx, row in tqdm(train_dataset.iterrows()):
    text = row['text']

    categories = re.findall('Category:', text)
    number_of_categories = len(categories)
    train_dataset['no_of_categories'][idx] = number_of_categories

    # regex to count find all external urls
    # urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    # number_of_external_links = len(urls)
    # train_dataset['no_external_links'][idx] = number_of_external_links
    loc = text.find('External links')
    if loc > 0:
        ntext = text[loc:]
        external_urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ntext)
        train_dataset['no_external_links'][idx] = len(external_urls)

    # internal links
    internal_links = re.findall('(?<=).*(?=)', text)
    number_of_internal_links = 0
    for links in internal_links:
        if ':' in links:
            continue
        else:
            number_of_internal_links += 1
    train_dataset['no_internal_links'][idx] = number_of_internal_links

    # level 2 and level 3+ headings
    all_headings = re.findall('\=\=+.+?\=\=+', text)
    no_of_level_2 = 0
    no_of_level_3_plus = 0

    # print(heading.count('='))
    for heading in all_headings:
        if (heading.count('=') // 2) == 2:
            no_of_level_2 += 1
        else:
            no_of_level_3_plus += 1
    
    train_dataset['no_level_2_headers'][idx] = no_of_level_2
    train_dataset['no_level_3_plus_headers'][idx] = no_of_level_3_plus

    # Number of references
    count = 0
    count += text.count('Reflist')
    count += text.count('reflist')
    for t in re.findall('refbegin([\S\s]*?)refend', text):
        count += t.count('*')
    train_dataset['no_of_references'][idx] = count


    # Number of images in text
    no_of_imgs = 0
    no_of_files = 0
    no_of_imgs = text.count('[File')
    no_of_files = text.count('|Image')
    train_dataset['no_of_images'][idx] = no_of_imgs + no_of_files

train_dataset.to_csv('./dataset/train_dataset_basic_ft.csv', index=False)

# Test dataset structural features
test_dataset['article_length_bytes'] = test_dataset['text'].apply(lambda s: len(s.encode('utf-8')))
test_dataset['no_of_citations'] = test_dataset['text'].apply(lambda s: s.count('<ref>'))
test_dataset['no_of_non_citations'] = test_dataset['text'].apply(lambda s: s.count('{{cn}}'))
test_dataset['no_of_infobox'] = test_dataset['text'].apply(lambda s: s.count('{{Infobox'))
test_dataset['no_of_categories'] = 0
test_dataset['no_external_links'] = 0
test_dataset['no_internal_links'] = 0
test_dataset['no_level_2_headers'] = 0
test_dataset['no_level_3_plus_headers'] = 0
test_dataset['no_of_images'] = 0
test_dataset['no_of_references'] = 0

for idx, row in tqdm(test_dataset.iterrows()):
    text = row['text']

    categories = re.findall('Category:', text)
    number_of_categories = len(categories)
    test_dataset['no_of_categories'][idx] = number_of_categories

    # regex to count find all external urls
    # urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    # number_of_external_links = len(urls)
    # test_dataset['no_external_links'][idx] = number_of_external_links
    loc = text.find('External links')
    if loc > 0:
        ntext = text[loc:]
        external_urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ntext)
        test_dataset['no_external_links'][idx] = len(external_urls)

    # internal links
    internal_links = re.findall('(?<=).*(?=)', text)
    number_of_internal_links = 0
    for links in internal_links:
        if ':' in links:
            continue
        else:
            number_of_internal_links += 1
    test_dataset['no_internal_links'][idx] = number_of_internal_links

    # level 2 and level 3+ headings
    all_headings = re.findall('\=\=+.+?\=\=+', text)
    no_of_level_2 = 0
    no_of_level_3_plus = 0

    for heading in all_headings:
        if (heading.count('=') // 2) == 2:
            no_of_level_2 += 1
        else:
            no_of_level_3_plus += 1
    
    test_dataset['no_level_2_headers'][idx] = no_of_level_2
    test_dataset['no_level_3_plus_headers'][idx] = no_of_level_3_plus

    # Number of images in text
    no_of_imgs = 0
    no_of_files = 0
    no_of_imgs = text.count('[File')
    no_of_files = text.count('|Image')
    test_dataset['no_of_images'][idx] = no_of_imgs + no_of_files

    # Number of references
    count = 0
    count += text.count('Reflist')
    count += text.count('reflist')
    for t in re.findall('refbegin([\S\s]*?)refend', text):
        count += t.count('*')
    test_dataset['no_of_references'][idx] = count

test_dataset.to_csv('./dataset/test_dataset_basic_ft.csv', index=False)