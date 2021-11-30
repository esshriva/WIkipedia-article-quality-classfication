import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import textstat
import mwparserfromhell
pd.set_option('chained_assignment', None)
import nltk

def isPassive(sentence):
    beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']               # all forms of "be"
    aux = ['do', 'did', 'does', 'have', 'has', 'had']                                  # NLTK tags "do" and "have" as verbs, which can be misleading in the following section.
    words = nltk.word_tokenize(sentence)
    tokens = nltk.pos_tag(words)
    tags = [i[1] for i in tokens]
    if tags.count('VBN') == 0:                                                            # no PP, no passive voice.
        return False
    elif tags.count('VBN') == 1 and 'been' in words:                                    # one PP "been", still no passive voice.
        return False
    else:
        pos = [i for i in range(len(tags)) if tags[i] == 'VBN' and words[i] != 'been']  # gather all the PPs that are not "been".
        for end in pos:
            chunk = tags[:end]
            start = 0
            for i in range(len(chunk), 0, -1):
                last = chunk.pop()
                if last == 'NN' or last == 'PRP':
                    start = i                                                             # get the chunk between PP and the previous NN or PRP (which in most cases are subjects)
                    break
            sentchunk = words[start:end]
            tagschunk = tags[start:end]
            verbspos = [i for i in range(len(tagschunk)) if tagschunk[i].startswith('V')] # get all the verbs in between
            if verbspos != []:                                                            # if there are no verbs in between, it's not passive
                for i in verbspos:
                    if sentchunk[i].lower() not in beforms and sentchunk[i].lower() not in aux:  # check if they are all forms of "be" or auxiliaries such as "do" or "have".
                        break
                else:
                    return True
    return False

def get_passive_sent_count(text):
    sents = nltk.sent_tokenize(text)
    cnt = 0
    for sent in sents:
        if isPassive(sent):
            cnt += 1
    return cnt

def get_number_of_questions(text):
    return text.count('?')

def get_number_of_aux_verbs(text):
    aux_verbs = ['be', 'am', 'are', 'is', 'was', 'were', 'being','can','could','do', 'did', 'does', 'doing',
                 'have','had', 'has', 'having','may','might','must','shall','should','will','would']
    tokens = nltk.word_tokenize(text)
    cnt = 0
    for aux_verb in aux_verbs:
        cnt += tokens.count(aux_verb)
    return cnt

def get_conjunction_rate(text):
    tokens = nltk.word_tokenize(text)
    tags = list(filter(lambda x: x[1] == 'CC', nltk.pos_tag(tokens)))
    return len(tags) / len(tokens)

def get_pronouns_count(text):
    tokens = nltk.word_tokenize(text)
    pronouns = list(filter(lambda x: x[1] in ['PRP', 'PRP$', 'WP', 'WP$'], nltk.pos_tag(tokens)))
    return len(pronouns)

def get_prepositions_rate(text):
    tokens = nltk.word_tokenize(text)
    prep = list(filter(lambda x: x[1] == 'IN', nltk.pos_tag(tokens)))
    return len(prep) / len(tokens)

def get_tobe_verb_percent(text):
    tobe_words = ['am', 'is', 'are', 'was', 'were', 'being', 'be', 'been']
    verbs_type = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    tokens = nltk.word_tokenize(text)

    to_be_cnt = 0
    vert_cnt = 0

    bigrm = list(nltk.bigrams(clean_text.split()))
    to_be_cnt += len(list(filter(lambda x:x == ('to', 'be'), bigrm)))

    to_be_cnt += len(list(filter(lambda x: x in tobe_words, tokens)))

    verb_cnt = len(list(filter(lambda x: x[0] not in tobe_words and x[1] in verbs_type, nltk.pos_tag(tokens))))

    try:
        return (to_be_cnt / verb_cnt) * 100.0
    except Exception:
        return 0

def get_sent_with_pronoun_as_beginning(text):
    sents = nltk.sent_tokenize(text)
    cnt = 0
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        pos_ = nltk.pos_tag(tokens)
        if pos_[0][1] in ['PRP', 'PRP$', 'WP', 'WP$']:
            cnt += 1
        
    return cnt

def get_sent_with_article_as_beginning(text):
    sents = nltk.sent_tokenize(text)
    cnt = 0
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        if tokens[0].lower() in ['a', 'an', 'the']:
            cnt += 1
        
    return cnt

def get_sent_with_conj_as_beginning(text):
    sents = nltk.sent_tokenize(text)
    cnt = 0
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        pos_ = nltk.pos_tag(tokens)
        if pos_[0][1] == 'CC':
            cnt += 1
        
    return cnt

def get_sent_with_subordinate_conj_as_beginning(text):
    sents = nltk.sent_tokenize(text)
    cnt = 0
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        pos_ = nltk.pos_tag(tokens)
        if pos_[0][1] == 'IN':
            cnt += 1
        
    return cnt

def get_sent_with_interrogative_pronoun_as_beginning(text):
    sents = nltk.sent_tokenize(text)
    cnt = 0
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        if tokens[0].lower() in ['who', 'whom', 'whose', 'what', 'which']:
            cnt += 1
        
    return cnt

def apply_on_train_dataset():
    # basic pipeline
    train_dataset = pd.read_csv('./dataset/train_dataset_basic_ft.csv', usecols=['text', 'labels'])
    train_dataset['clean_text'] = train_dataset['text'].apply(lambda text: re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text))

    new_df = pd.DataFrame(columns=['id', 'PassiveSent', 'Ques', 'AuxVerbs', 'ConjRate', 'Pronouns', 'PreposRate', 'ToBeRate', 'ProAtB', 'ArticleAtB', 'ConjAtB', 'SubConjAtB', 'IntProAtB'])

    dirty_text = train_dataset['clean_text'].tolist()

    for i in tqdm(range(len(dirty_text))):
        feat = []

        text = dirty_text[i]

        wikicode = mwparserfromhell.parse(text)
        clean_text = ' '.join(list(map(str, wikicode.filter_text())))

        feat.append(i)
        feat.append(get_passive_sent_count(clean_text))
        feat.append(get_number_of_questions(clean_text))
        feat.append(get_number_of_aux_verbs(clean_text))
        feat.append(get_conjunction_rate(clean_text))
        feat.append(get_pronouns_count(clean_text))
        feat.append(get_prepositions_rate(clean_text))
        feat.append(get_tobe_verb_percent(clean_text))
        feat.append(get_sent_with_pronoun_as_beginning(clean_text))
        feat.append(get_sent_with_article_as_beginning(clean_text))
        feat.append(get_sent_with_conj_as_beginning(clean_text))
        feat.append(get_sent_with_subordinate_conj_as_beginning(clean_text))
        feat.append(get_sent_with_interrogative_pronoun_as_beginning(clean_text))

        new_df.loc[new_df.shape[0]] = feat

    new_df['labels'] = train_dataset['labels']
    new_df.to_csv('./dataset/train_dataset_writing_style.csv', index=False)

def apply_on_test_dataset():
    test_dataset = pd.read_csv('./dataset/test_dataset_basic_ft.csv', usecols=['text', 'label'])
    test_dataset.columns = ['text', 'labels']
    test_dataset['clean_text'] = test_dataset['text'].apply(lambda text: re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text))

    new_df = pd.DataFrame(columns=['id', 'PassiveSent', 'Ques', 'AuxVerbs', 'ConjRate', 'Pronouns', 'PreposRate', 'ToBeRate', 'ProAtB', 'ArticleAtB', 'ConjAtB', 'SubConjAtB', 'IntProAtB'])

    dirty_text = test_dataset['clean_text'].tolist()

    for i in tqdm(range(len(dirty_text))):
        feat = []

        text = dirty_text[i]

        wikicode = mwparserfromhell.parse(text)
        clean_text = ' '.join(list(map(str, wikicode.filter_text())))

        feat.append(i)
        feat.append(get_passive_sent_count(clean_text))
        feat.append(get_number_of_questions(clean_text))
        feat.append(get_number_of_aux_verbs(clean_text))
        feat.append(get_conjunction_rate(clean_text))
        feat.append(get_pronouns_count(clean_text))
        feat.append(get_prepositions_rate(clean_text))
        feat.append(get_tobe_verb_percent(clean_text))
        feat.append(get_sent_with_pronoun_as_beginning(clean_text))
        feat.append(get_sent_with_article_as_beginning(clean_text))
        feat.append(get_sent_with_conj_as_beginning(clean_text))
        feat.append(get_sent_with_subordinate_conj_as_beginning(clean_text))
        feat.append(get_sent_with_interrogative_pronoun_as_beginning(clean_text))

        new_df.loc[new_df.shape[0]] = feat

    new_df['labels'] = test_dataset['labels']
    new_df.to_csv('./dataset/test_dataset_writing_style.csv', index=False)

if __name__ == '__main__':
    # apply_on_train_dataset()
    # apply_on_test_dataset()
    pass
