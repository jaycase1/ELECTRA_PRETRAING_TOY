import keras.backend as K
from utils import read_json,maskID
from keras.layers import Layer,Dense,Input
import tensorflow as tf
from keras.models import Model
import keras
from bert4keras.models import BERT,build_transformer_model

class argmaxLambda(Layer):
    def call(self, inputs, **kwargs):
        return tf.argmax(inputs,axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

class GEN_ELECTRA(Layer):
    def __init__(self,config_path="generator_config.json",maskID=maskID,penalty_factor=2,**kwargs):
        super(GEN_ELECTRA,self).__init__(**kwargs)
        self.Bert = build_transformer_model(config_path=config_path)
        self.vocab_size = read_json(config_path)["vocab_size"]
        self.maskID = maskID
        self.penalty_factor = penalty_factor

    def build(self, input_shape):
        super(GEN_ELECTRA,self).build(input_shape)
        self.predict_ = Dense(units=self.vocab_size,activation="softmax")

    def sample(self,output,ascertain=True):
         if(ascertain):
             return K.argmax(output, axis=-1)
         else:
             # TODO: 按照概率取值
             pass

    def call(self, inputs, **kwargs):
        output = self.Bert(inputs)
        output = self.predict_(output)
        #output = self.sample(output)
        return output

    def compute_mask(self, inputs, mask=None):
        if(mask is not None):
            #print(43,mask)
            return mask


loss_object = keras.losses.SparseCategoricalCrossentropy(
    # from_logits=True,  # 未经过softmax的结果
    reduction='none'  # 禁止求和计算，这一步在自定义的loss中做
)


def loss_func(y_true, y_pred,mask):
    #mask = tf.math.logical_not(tf.math.equal(y_true, maskID))  # 将y_true 中所有为0的找出来，标记为False
    loss_ = loss_object(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)  # 将前面统计的是否零转换成1，0的矩阵
    loss_ *= mask  # 将正常计算的loss加上mask的权重，就剔除了padding 0的影响
    return tf.reduce_mean(loss_)

def cal_GEN_loss(inputs,corrupt,outputs,maskID=maskID,penalty_factor=2.0):
    #corrupt = inputs
    mask_ = (corrupt  == maskID)
    not_mask_ = (corrupt != maskID)
    loss = loss_func(inputs,outputs,mask_)
    loss += penalty_factor * loss_func(inputs,outputs,not_mask_)
    return loss


"""
x = Input(shape=(None,))
y = Input(shape=(None,))
z = GEN_ELECTRA()([x,y])

model = Model([x,y],z)
model.add_loss(cal_GEN_loss([x,y],z))
model.summary()
"""

