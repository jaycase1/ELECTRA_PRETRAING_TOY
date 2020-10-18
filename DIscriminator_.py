import keras.backend as K
from utils import read_json,maskID
from keras.layers import Layer,Dense,Input
import tensorflow as tf
from keras.models import Model
import keras
from bert4keras.models import BERT,build_transformer_model

class DIS_ELECTRA(Layer):
    def __init__(self, config_path="discriminator_config.json", maskID=maskID, penalty_factor=2, **kwargs):
        super(DIS_ELECTRA,self).__init__(**kwargs)
        self.Bert_1 = build_transformer_model(config_path=config_path)
        self.maskID = maskID
        self.penalty_factor = penalty_factor


    def build(self, input_shape):
        super(DIS_ELECTRA,self).build(input_shape)
        self.predict_ = Dense(2,activation="sigmoid")

    def call(self, inputs, mask=None):
        """
        :param inputs:  a list of  tensor , the  position 1 is Bert_GEN's output
        :param kwargs:
        :return:
        """
        if(mask is not None):
            inputs = inputs * mask
        output = self.Bert_1(inputs)
        print(30)
        output = self.predict_(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][:2])+(2,)

    def compute_mask(self, inputs, mask=None):
        if(mask is not None):
            print(mask ,36)
            return mask


"""

x = Input(shape=(None,))
y = Input(shape=(None,))

z = DIS_ELECTRA()([x,y])
print(z)
"""

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

def cal_DIS_Loss(masked,outputs,labels):
    """
    :param inputs: 原始文本序列
    :param corrupts: 这个是 Generator 生成的序列
    :param outputs:  判别器判断出的 mask_or_not 序列
    :param labels: {1,0} 组成的Tensor 1表示这个词没有被mask掉
                                    0 表示这个词被mask掉了
    :return:
    """
    """
    masked = (labels == 1)
    correct = (inputs == corrupts)
    masked = tf.cast(masked,tf.float32)
    correct = tf.cast(correct,tf.float32)
    mask = masked * correct
    """
    return loss_func(labels,outputs,masked)

"""
inputs =  Input(shape=(None,))
x = Input(shape=(None,))
y = Input(shape=(None,))
labels = Input(shape=(None,))

z = DIS_ELECTRA()([x,y])
DIS_Model = Model([x,y],z)
DIS_Model.summary()
DIS_Model.add_loss(cal_DIS_Loss(inputs=inputs,corrupts=x,outputs=z,labels=labels))
DIS_Model.summary()

"""



