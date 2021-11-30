import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle as pb
from tqdm import tqdm
from datetime import datetime

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

class ArticleClassifierDeep(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(ArticleClassifierDeep, self).__init__()
        self.linear1 = nn.Linear(input_shape, input_shape*2)
        self.batch_norm1 = nn.BatchNorm1d(input_shape*2)
        self.linear2 = nn.Linear(input_shape*2, input_shape*3)
        self.batch_norm2 = nn.BatchNorm1d(input_shape*3)
        self.linear3 = nn.Linear(input_shape*3, input_shape*2)
        self.batch_norm3 = nn.BatchNorm1d(input_shape*2)
        self.linear4 = nn.Linear(input_shape*2, input_shape)
        self.batch_norm4 = nn.BatchNorm1d(input_shape)
        self.final = nn.Linear(input_shape, n_classes)
        self.dropout1 = nn.Dropout(0.5)
        # self.bert = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
    
    def forward(self, x):
        x = self.batch_norm1(F.relu(self.linear1(x)))
        x = self.dropout1(x)
        x = self.batch_norm2(F.relu(self.linear2(x)))
        x = self.dropout1(x)
        x = self.batch_norm3(F.relu(self.linear3(x)))
        x = self.dropout1(x)
        x = self.batch_norm4(F.relu(self.linear4(x)))
        x = self.final(x)
        # x = self.bert(input_ids=x, labels=labels)
        return x

if __name__ == '__main__':
    # Training Dataset Preparation
    df = pd.read_csv('./dataset/train_dataset_token_len.csv', usecols=['text', 'labels'])

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
    model = ArticleClassifier(dataset[0][0].shape[0], 6).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    NUM_EPOCHS = 50
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    lowest_loss = 999

    for e in range(NUM_EPOCHS):
        epoch_loss = []
        epoch_val_loss = []
        correct = 0
        total = 0
        val_correct = 0
        val_total = 0
        start = datetime.now()
        model.train()
        for step, (inputs, labels) in tqdm(enumerate(dataloader), desc=f'Training Epoch-{e+1}'):
            optimizer.zero_grad()

            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))

            epoch_loss.append(loss.item())

            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels.to(device)).sum().item()

            loss.backward()
            optimizer.step()
        model.eval()
        for step, (inputs, labels) in tqdm(enumerate(test_dataloader), desc=f'Validation Epoch-{e+1}'):
            with torch.no_grad():
                outputs = model(inputs.to(device))
                loss = loss_fn(outputs, labels.to(device))
                epoch_val_loss.append(loss.item())

                _, preds = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels.to(device)).sum().item()

        end = datetime.now()

        acc = (correct / total) * 100.0
        v_acc = (val_correct / val_total) * 100.0

        train_acc.append(acc)
        val_acc.append(v_acc)

        train_loss.append(np.mean(epoch_loss))
        val_loss.append(np.mean(epoch_val_loss))

        if np.mean(epoch_val_loss) < lowest_loss:
            lowest_loss = np.mean(epoch_val_loss)
            print('Best Model Saved at Epoch:', e+1)
            torch.save(model, 'best_article_classifier.pt')

        print(f"Epoch {e+1}/{NUM_EPOCHS}, Time-Taken:{end-start}, Train-Loss:{np.mean(epoch_loss):.4f}, Train-Acc:{acc:.4f}, Validation-Loss:{np.mean(epoch_val_loss):.4f}, Validation-Acc:{v_acc:.4f}")
    
    # Save Model
    torch.save(model, 'article-classifier.pt')

    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.savefig('training_loss.png')
    plt.clf()

    plt.plot(train_acc)
    plt.title('Train Accuracy')
    plt.savefig('training_acc.png')
    plt.clf()

    plt.plot(val_loss)
    plt.title('Validation Loss')
    plt.savefig('validation_loss.png')
    plt.clf()

    plt.plot(val_acc)
    plt.title('Validation Accuracy')
    plt.savefig('validation_acc.png')
    plt.clf()



