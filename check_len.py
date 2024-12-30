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

import matplotlib.pyplot as plt

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

def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = wordpunct_tokenize(text)
    #tokens = tokens[:-1] # remove last token because it is the year which maybe is not useful
    return tokens

list_title = [tokenize(x) for x in movies_train.title]

length_distribution = [len(title) for title in list_title]

print("Length distribution of movie titles:")
for length, count in zip(sorted(set(length_distribution)), [length_distribution.count(l) for l in sorted(set(length_distribution))]):
    print(f"Length {length}: {count} titles")

# Alternatively, if you want to see the lengths and titles together
print("\nMovie titles with their lengths:")
# for title, length in zip(list_title, length_distribution):
#     print(f"{title} - Length: {length}")


plt.bar(sorted(set(length_distribution)), [length_distribution.count(l) for l in sorted(set(length_distribution))])
plt.xlabel('Title Length')
plt.ylabel('Number of Titles')
plt.title('Length Distribution of Movie Titles')

# Save the chart as an image
plt.savefig('length_distribution_chart.png')

# Display the chart
plt.show()