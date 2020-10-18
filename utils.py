from bert4keras.tokenizers import Tokenizer , load_vocab
import json
import numpy as np
dict_path = "vocab.txt"

tokenizer = Tokenizer(load_vocab(dict_path))
maskID = tokenizer.token_to_id(tokenizer._token_mask)



def write_Json(content,fileName):
    with open(fileName,"w") as f:
        json.dump(content,f,indent=2)


def read_json(fileName):
    fp = open(fileName,"r")
    f = json.load(fp)
    return f


def cal_mask(inputs,corrupts,labels):
    assert inputs.shape == corrupts.shape and corrupts.shape == labels.shape
    masked = (labels == 1)
    correct = (inputs == corrupts)
    masked = masked.astype(np.float)
    correct = correct.astype(np.float)
    mask = masked * correct
    return mask