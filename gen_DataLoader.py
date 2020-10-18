from math import ceil
import numpy as np
import copy

class GEN_DataLoader(object):
    def __init__(self,data,tokenizer,batchSize=16,mask_Rate=0.15,maxLen=100):
        self.data = data
        self.tokenizer = tokenizer
        self.padID = tokenizer.token_to_id(tokenizer._token_pad)
        self.maskID = tokenizer.token_to_id(tokenizer._token_mask)
        self.steps = ceil(float(len(data))/batchSize)
        self.maxLen = maxLen
        self.batchSize = batchSize
        self.mask_Rate = mask_Rate
        self.ptr = 0
        self.idxs = None

    def __len__(self):
        return self.steps

    def seq_padding(self,X):
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([
            np.concatenate([x, [self.padID] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])

    def mask_range(self,bool_mat,bound):
        #mask_Ids = []
        flag = 0
        for len_ in bound:
            #print("random choice, ", np.random.choice(len_, ceil(len_ * self.mask_Rate)))
            idxs = np.random.choice(len_ - 1, ceil(len_ * self.mask_Rate))
            for id in idxs:
                bool_mat[(flag,id)] = True
            flag += 1
        return bool_mat


    def seq_masking(self,Inputs):
        elementsLen = (Inputs != self.padID).sum(axis=-1)
        boolMat = np.zeros_like(Inputs, dtype=np.bool)
        boolMat = self.mask_range(boolMat,elementsLen)
        Inputs[boolMat] = self.maskID
        return Inputs,boolMat

    def __next__(self):
        if(self.ptr<self.__len__()):
            X = []
            X_ = []
            if(self.ptr<self.__len__()-1):
                idxs = self.idxs[self.ptr * self.batchSize: (self.ptr + 1)*self.batchSize]
            else:
                idxs = self.idxs[self.ptr * self.batchSize :]
            for i in idxs:
                d = self.data[i][0]
                text = d[:self.maxLen]
                x1, x2 = self.tokenizer.encode(text)
                X.append(x1)
                X_.append(x2)
            X = self.seq_padding(X)
            origin_ = copy.deepcopy(X)
            X_ = self.seq_padding(X_)
            assert X.shape == X_.shape
            X, labels = self.seq_masking(X)
            labels = labels + 0
            self.ptr += 1
            return {"origin_ids":origin_,"mask_id":X,"sentences_id":X_,"mask_labels":labels}


        else:
            self.ptr = 0
            print("end"
                )
            raise StopIteration

    def __iter__(self):
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        self.idxs = idxs
        while(True):
            try:
                yield self.__next__()
            except:
                break
