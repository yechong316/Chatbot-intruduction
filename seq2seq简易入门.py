# coding:utf-8
'''
通过认为设计一个训练样本，输入N个句子，返回N个句子，输入至tf中，逐步观察各个out是什么东西，加深对
tf的理解

'''


import numpy as np

# 输入序列长度
input_seq_len = 5
# 输出序列长度
output_seq_len = 5
# 空值填充0
PAD_ID = 0
# 输出序列起始标记
GO_ID = 1


def get_samples():
    """构造样本数据

    :return:
        encoder_inputs: [array([0, 0], dtype=int32),
                         array([0, 0], dtype=int32),
                         array([1, 3], dtype=int32),
                         array([3, 5], dtype=int32),
                         array([5, 7], dtype=int32)]
        decoder_inputs: [array([1, 1], dtype=int32),
                         array([7, 9], dtype=int32),
                         array([ 9, 11], dtype=int32),
                         array([11, 13], dtype=int32),
                         array([0, 0], dtype=int32)]
    """
    train_set = [
        [
            [1, 3, 5], [7, 9, 11]
        ],
        [
            [3, 5, 7], [9, 11, 13]
        ],
        [
            [5, 7, 9], [11,13, 15]
        ],
    ]
    encoder_inputs_ = [
        [PAD_ID] * (input_seq_len - len(train_set[i][0])) + train_set[i][0] for i in range(len(train_set))
    ]
    decoder_inputs_ = [
        [GO_ID] + train_set[i][1] + [PAD_ID] * (output_seq_len - len(train_set[i][1]) - 1) for i in range(len(train_set))
    ]

    encoder_inputs = [
        np.array([encoder_inputs_[j][i] for j in range(len(train_set))], dtype=np.int32)  for i in range(input_seq_len)
    ]
    
    decoder_inputs = [
        np.array([decoder_inputs_[j][i] for j in range(len(train_set))], dtype=np.int32)  for i in range(input_seq_len)
    ]



    return encoder_inputs, decoder_inputs


import tensorflow as tf
encoder_inputs = []
decoder_inputs = []
for i in range(input_seq_len):
    encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                          name="encoder{0}".format(i)))
for i in range(output_seq_len):
    decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                          name="decoder{0}".format(i)))


size = 32
cell = tf.contrib.rnn.BasicLSTMCell(size)
num_encoder_symbols = 10
num_decoder_symbols = 10


def get_model():
    """构造模型
    """
    encoder_inputs = [
        tf.placeholder(tf.int32, [None], 'encoder{0}'.format(i)) for i in range(input_seq_len)
    ]
    decoder_inputs = [
        tf.placeholder(tf.int32, [None], 'decoder{0}'.format(i)) for i in range(output_seq_len)
    ]

    cell = tf.contrib.rnn.BasicLSTMCell(size)

    # 这里输出的状态我们不需要
    outputs, hidden = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                        encoder_inputs,
                        decoder_inputs,
                        cell,
                        num_encoder_symbols=num_encoder_symbols,
                        num_decoder_symbols=num_decoder_symbols,
                        embedding_size=size,
                        output_projection=None,
                        feed_previous=False,
                        dtype=tf.float32)
    return encoder_inputs, decoder_inputs, outputs, hidden


samples_encoder_inputs, samples_decoder_inputs = get_samples()
# print(samples_encoder_inputs, samples_decoder_inputs)

encoder_inputs, decoder_inputs, outputs, hidden = get_model()
with tf.Session() as sess:

    infeed = {}
    for key, value in zip(encoder_inputs, samples_encoder_inputs):
        infeed[key.name] = value
    for key, value in zip(decoder_inputs, samples_decoder_inputs):
        infeed[key.name] = value
    sess.run(tf.global_variables_initializer())
    out, h = sess.run([outputs, hidden], infeed)

    # print('out.length:', len(out))

    for i in range(len(out)):
        print('out[{}].shape = {}'.format(i, out[i].shape))
    # print('h.length:', len(h))h
    for i in range(len(h)):
        print('h[{}].shape = {}'.format(i, h[i].shape))

# encoder_name = [i.name for i in encoder_inputs]
# decoder_name = [i.name for i in decoder_inputs]
# # for i in range(len(encoder_name)):
# #
# #     # print('encoder_name: ', encoder_name[i])
# #     print('samples_encoder_inputs: ', samples_encoder_inputs[i].shape)
# #     print('samples_encoder_inputs: ', samples_encoder_inputs[i])
# for i in range(len(samples_decoder_inputs)):
#     print('decoder_name: ', samples_decoder_inputs[i])