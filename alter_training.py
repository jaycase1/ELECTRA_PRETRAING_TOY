from generator_ import GEN_ELECTRA , cal_GEN_loss
from DIscriminator_ import DIS_ELECTRA, cal_DIS_Loss
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from dataLoader_test import train_G
import numpy as  np
from Dis_DataLoader import DIS_DataLoader
from utils import cal_mask
import copy

lr = 1e-4

lr_rate = 5

GEN = GEN_ELECTRA()
DIS = DIS_ELECTRA()

# ========================================================
# 生成模型定义

origin_inputs = Input(shape=(None,))
mask_ids = Input(shape=(None,))
segments_ids = Input(shape=(None,))

predict_Pro = GEN([mask_ids,segments_ids])

model_GEN = Model([origin_inputs,mask_ids,segments_ids],predict_Pro)
model_GEN.add_loss(cal_GEN_loss(inputs=origin_inputs,corrupt=mask_ids,outputs=predict_Pro))
model_GEN.compile(
    optimizer=Adam(lr), # 用足够小的学习率
    metrics=['accuracy']
)

# ==================================================
# 判别模型定义
mask_ = Input(shape=(None,))
gen_Input = Input(shape=(None,))
segments_ids_ = Input(shape=(None,))
labels = Input(shape=(None,))
DIS = DIS_ELECTRA()
Dis_out = DIS([gen_Input,segments_ids_])
model_DIS = Model([mask_,gen_Input,segments_ids_,labels],Dis_out)
model_DIS.add_loss(
    cal_DIS_Loss(mask_,outputs=Dis_out,labels=labels)
)
model_DIS.compile(
    optimizer=Adam(lr*lr_rate), # 用足够小的学习率
    metrics=['accuracy']
)


Epochs = 1
vs_Steps = 1


for epoch in range(Epochs):
    train_D = DIS_DataLoader()
    print("start Traing")
    i = 0
    for d_gen in train_G:
        i += 1
        if(i==20):
            break
        print(i,"/",train_G.__len__())
        model_GEN.train_on_batch([d_gen["origin_ids"],d_gen["mask_id"],d_gen["sentences_id"]],None)
        gen_output = model_GEN.predict([d_gen["origin_ids"],d_gen["mask_id"],d_gen["sentences_id"]])
        gen_output = np.argmax(gen_output,axis=-1)
        #print(i ,"/",train_G.__len__())
        print(gen_output.shape)
        train_D.add_element([cal_mask(d_gen["origin_ids"],np.array(gen_output.data),d_gen["mask_labels"]),gen_output,d_gen["sentences_id"],d_gen["mask_labels"]])
    print("starting Dis")
    for i in range(vs_Steps):
        j = 0
        for d in train_D:
            j += 1
            print(j,"/",train_D.__len__())

            # 可以预测但不能train 定义梯度无法传播运算
            model_DIS.train_on_batch(d,None)
            pre = model_DIS.predict(d)

            print(pre.shape)

