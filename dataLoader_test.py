import pandas as pd
import numpy as np
from gen_DataLoader import GEN_DataLoader
from bert4keras.tokenizers import Tokenizer
import codecs

dict_path = "vocab.txt"


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

neg = pd.read_excel('neg.xls', header=None)
pos = pd.read_excel('pos.xls', header=None)

data = []

for d in neg[0]:
    data.append((d, 0))

for d in pos[0]:
    data.append((d, 1))

random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

train_G = GEN_DataLoader(
    train_data,tokenizer
)
