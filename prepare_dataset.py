import pandas as pd
pd.set_option('chained_assignment', None)

train_df = pd.read_csv("./datasets/training-set.tsv", sep='\t')
test_df = pd.read_csv('./datasets/test-set.tsv', sep='\t')

train_dataset = pd.DataFrame(data=[], columns=['text', 'labels'])
test_dataset = pd.DataFrame(data=[], columns=['text', 'labels'])
for idx, row in train_df.iterrows():
    revid = row['revid']
    with open(f'./revisiondata/{revid}', 'r') as f:
        data = f.read()
    train_dataset.loc[train_dataset.shape[0]] = [data, row['ordered_class']]

for idx, row in test_df.iterrows():
    revid = row['revid']
    with open(f'./revisiondata/{revid}', 'r') as f:
        data = f.read()
    test_dataset.loc[test_dataset.shape[0]] = [data, row['ordered_class']]

train_dataset.to_csv('./dataset/training_set.csv', index=False)
test_dataset.to_csv('./dataset/testing_set.csv', index=False)