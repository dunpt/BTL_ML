import pandas
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import cv2
import os
from nltk import wordpunct_tokenize
import re
from dataloader import MLDataset, create_vocab
from model import BaseModel
# You need extract file ml1m.zip to folder ml1m before run code

users = pandas.read_csv('./dataset/users.dat', sep='::',
                        engine='python',
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')
ratings = pandas.read_csv('./dataset/ratings.dat', engine='python',
                          sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
movies_train = pandas.read_csv('./dataset/movies_train.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
movies_test = pandas.read_csv('./dataset/movies_test.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')

movies_train['genre'] = movies_train.genre.str.split('|')
movies_test['genre'] = movies_test.genre.str.split('|')

users.age = users.age.astype('category')
users.gender = users.gender.astype('category')
users.occupation = users.occupation.astype('category')
ratings.movieid = ratings.movieid.astype('category')
ratings.userid = ratings.userid.astype('category')

#print(movies_train)

folder_img_path = './dataset/ml1m-images'
movies_train['id'] = movies_train.index
movies_train.reset_index(inplace=True)
movies_train['img_path'] = movies_train.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)
#print(movies_train['title'])

folder_img_path = './dataset/ml1m-images'
movies_test['id'] = movies_test.index
movies_test.reset_index(inplace=True)
movies_test['img_path'] = movies_test.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)
#print(movies_test)

def check(row):
  print(row)
  return not os.path.exists(row["img_path"])

def get_dataframe(dataframe):
  mask = dataframe.apply(check, axis=1)

  data = dataframe.drop(dataframe[mask].index)
  return data

movies_train = get_dataframe(movies_train)
movies_test = get_dataframe(movies_test)
print(len(movies_train))

train_set = MLDataset(movies_train, is_train=True)
test_set = MLDataset(movies_test, is_train=False)



BATCH_SIZE = 8
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE)

#####
def buildAdjacencyCOOC(data_label):
  adj = data_label.T.dot(data_label).astype('float')
  for i in range(len(adj)):
    adj[i] = adj[i] / adj[i,i]
  return torch.from_numpy(adj.astype('float32'))

def loadWRVModel(File):
    print("Loading Word Representation Vector Model")
    f = open(File,'r')
    WRVModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        try:
          wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        except:
          print(splitLines[1:])
          print(len(splitLines[1:]))
          break
        WRVModel[word] = wordEmbedding
    print(len(WRVModel)," words loaded!")
    return WRVModel

import pickle

with open('./multilabelbinarizer.pickle', 'rb') as file:
  mlb = pickle.load(file)

WRVModel = loadWRVModel('./glove.6B.300d.txt')

VOCAB_SIZE = len(tokenizer.word_index) + 1
embedding_matrix = torch.zeros(VOCAB_SIZE, 300)

unk = 0
for i in range(1, VOCAB_SIZE):
  word = tokenizer.index_word[i]
  if word in WRVModel.keys():
    embedding_matrix[i] = torch.from_numpy(WRVModel[word]).float()
  else:
    unk +=1
print('VOCAB_SIZE : {}'.format(VOCAB_SIZE))
print('TOTAL OF UNKNOWN WORD : {}'.format(unk))


label_embedding = torch.zeros(90,300)

for index, label in enumerate(mlb.classes_):
  words = label.split('-')
  num_of_words = len(words)

  for sublabel in words:
    if sublabel in WRVModel.keys():
      label_embedding[index] +=  torch.from_numpy(WRVModel[sublabel])
  label_embedding[index] = label_embedding[index]/num_of_words

print(label_embedding)

adjacency = buildAdjacencyCOOC(y_train.numpy())
print(adjacency)

#####

with open('./dataset/genres.txt', 'r') as f:
    genre_all = f.readlines()
num_classes = len(genre_all)
#print(num_classes)



vocab = create_vocab(movies_train)
#print("vocab", vocab)
model = BaseModel(num_classes, len(vocab))

# out = model(title_tensor, img_tensor)
# print(out.shape)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

count_parameters(model)

from torch import optim
criterion = nn.CrossEntropyLoss()

learning_rate = 1e-3
# optimizer = optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=learning_rate,
# )
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

device = torch.device('cuda:0')
NUM_EP = 10
model.to(device)
for ep in range(NUM_EP):

    print("="*50)
    for idx, (title_tensor, img_tensor, genre_tensor) in enumerate(train_dataloader):
        title_tensor = title_tensor.to(device)
        img_tensor = img_tensor.to(device)
        genre_tensor = genre_tensor.to(device)
        #print("title_tensor", title_tensor.shape, title_tensor)

        optimizer.zero_grad()
        out = model(title_tensor, img_tensor)

        loss = criterion(out, genre_tensor)
        if idx % 50 == 0 and idx > 0:
          print("loss: ", loss)
        loss.backward()
        
        optimizer.step()

from torchmetrics.classification import MultilabelF1Score
N, C = genre_tensor.shape

auroc_all = 0
f1_all = 0
f1 = MultilabelF1Score(num_labels=C, threshold=0.5, average='macro')
f1 = f1.to(device)
for title_tensor, img_tensor, genre_tensor in test_dataloader:
    title_tensor = title_tensor.to(device)
    img_tensor = img_tensor.to(device)
    genre_tensor = genre_tensor.to(device)
    print("title_tensor", title_tensor)
    out = model(title_tensor, img_tensor)
    f1_batch = f1(out, genre_tensor)
    f1_all += f1_batch

print('F1: ', f1_all/len(test_dataloader))
