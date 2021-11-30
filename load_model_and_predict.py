import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pb
import pandas as pd
pd.set_option('chained_assignment', None)
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, vec_files=[]):
        self.df = df
        self.vectors = []
        if len(vec_files) == 3:
            self.v1, self.v2, self.v3 = vec_files

            self.vectors.extend(self.v1)
            self.vectors.extend(self.v2)
            self.vectors.extend(self.v3)
        elif len(vec_files) == 1:
            self.vectors.extend(vec_files[0])
        else:
            raise "Either send 3 Vector list for train dataset or 1 vector list for test dataset"

        self.map_fn = {'Start':0, 'Stub':1, 'B':2, 'FA':3, 'GA':4, 'C':5}

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        label = self.map_fn[self.df.iloc[idx]['labels']]
        input_ft = torch.tensor(self.vectors[idx])

        return input_ft, label

class ArticleClassifier(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(ArticleClassifier, self).__init__()
        self.linear1 = nn.Linear(input_shape, input_shape*2)
        self.batch_norm1 = nn.BatchNorm1d(input_shape*2)
        self.linear2 = nn.Linear(input_shape*2, input_shape)
        self.batch_norm2 = nn.BatchNorm1d(input_shape)
        self.linear3 = nn.Linear(input_shape, n_classes)
        self.dropout1 = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.batch_norm1(F.relu(self.linear1(x)))
        x = self.dropout1(x)
        x = self.batch_norm2(F.relu(self.linear2(x)))
        x = self.dropout1(x)
        x = self.linear3(x)
        return x
    
if __name__ == '__main__':
    

    df = pd.read_csv('./dataset/train_dataset.csv', usecols=['text', 'labels'])

    # Load 0 to 8000 vector file
    with open('./old-train-test-embeddings/doc_embedding_0_8000.pt', 'rb') as f:
        vec_0_8000 = pb.load(f)

    # Load 8000 to 19000 vector file
    with open('./old-train-test-embeddings/doc_embedding_8000_19000.pt', 'rb') as f:
        vec_8000_19000 = pb.load(f)
    
    # Load 19000 to 26506 vector file
    with open('./old-train-test-embeddings/doc_embedding_19000_26506.pt', 'rb') as f:
        vec_19000_26506 = pb.load(f)
    
    print(len(vec_0_8000), len(vec_8000_19000), len(vec_19000_26506), df.shape)

    dataset = CustomDataset(df, vec_files=[vec_0_8000, vec_8000_19000, vec_19000_26506])
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=64, num_workers=0)

    # Test/Validation dataset preparation
    test_df = pd.read_csv('./dataset/test_dataset.csv', usecols=['text', 'label'])
    test_df.columns = ['text', 'labels']
    # Loading vector file
    with open('./old-train-test-embeddings/test_doc_embedding_0_2941.pt', 'rb') as f:
        vec_test = pb.load(f)['doc_embed']
    test_dataset = CustomDataset(test_df, vec_files=[vec_test])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=0)

    # Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = ArticleClassifier(dataset[0][0].shape[0], 6).to(device)

    model = torch.load('./65percent acc/best_article_classifier.pt').to(device)
    print(model)

    with torch.no_grad():
        train_preds = []
        train_labels = []
        val_preds = []
        val_labels = []

        total = 0
        correct = 0
        ## Accuracy on training set
        for step, (inputs, labels) in tqdm(enumerate(dataloader)):

            outputs = model(inputs.to(device))

            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels.to(device)).sum().item()
            train_preds.append(preds.cpu())
            train_labels.append(labels.cpu())
        print('Training Set Accuracy:', (correct / total) * 100.0)

        ## Accuracy on validation set
        val_total = 0
        val_correct = 0
        for step, (inputs, labels) in tqdm(enumerate(test_dataloader)):
            outputs = model(inputs.to(device))

            _, preds = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels.to(device)).sum().item()

            val_preds.append(preds.cpu())
            val_labels.append(labels.cpu())

        print('Validation Set Accuracy:', (val_correct / val_total) * 100.0)
    
    with open('output_preds_n_labels.pb', 'wb') as f:
        obj = {
            'train_preds': train_preds,
            'train_labels': train_labels,
            'val_preds': val_preds,
            'val_labels': val_labels
        }
        pb.dump(obj, f)

