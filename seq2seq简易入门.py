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
    # 给encoder进行PADDING
    encoder_inputs_ = [
        [PAD_ID] * (input_seq_len - len(train_set[i][0])) + train_set[i][0] for i in range(len(train_set))
    ]

    # 给encoder进行PADDING
    decoder_inputs_ = [
        [GO_ID] + train_set[i][1] + [PAD_ID] * (output_seq_len - len(train_set[i][1]) - 1) for i in range(len(train_set))
    ]

    # 给encoder进行转为维度， 构造为，词数 * batch数
    encoder_inputs = [
        np.array([encoder_inputs_[j][i] for j in range(len(train_set))], dtype=np.int32)  for i in range(input_seq_len)
    ]

    # 给encoder进行转为维度
    decoder_inputs = [
        np.array([decoder_inputs_[j][i] for j in range(len(train_set))], dtype=np.int32)  for i in range(output_seq_len)
    ]

    target_weights = [
        np.array([
            0.0 if length_idx == output_seq_len - 1 or decoder_inputs_[i][length_idx] == PAD_ID else 1.0
            for i in range(len(decoder_inputs_))
        ], dtype=np.float32) for length_idx in range(output_seq_len)
    ]


    return encoder_inputs, decoder_inputs, target_weights


import tensorflow as tf
encoder_inputs = []
decoder_inputs = []
for i in range(input_seq_len):
    encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                          name="encoder{0}".format(i)))
for i in range(output_seq_len + 1):
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
        tf.placeholder(tf.int32, [None], 'decoder{0}'.format(i)) for i in range(output_seq_len + 1)
    ]
    target_weights = [
        tf.placeholder(tf.float32, [None], 'weight{0}'.format(i)) for i in range(output_seq_len)
    ]

    cell = tf.contrib.rnn.BasicLSTMCell(size)

    # 这里输出的状态我们不需要
    outputs, hidden = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                        encoder_inputs,
                        decoder_inputs[:output_seq_len],
                        cell,
                        num_encoder_symbols=num_encoder_symbols,
                        num_decoder_symbols=num_decoder_symbols,
                        embedding_size=size,
                        output_projection=None,
                        feed_previous=False,
                        dtype=tf.float32)
    return encoder_inputs, decoder_inputs, target_weights, outputs, hidden


samples_encoder_inputs, samples_decoder_inputs, samples_target_weights = get_samples()


encoder_inputs, decoder_inputs, target_weights, outputs, hidden = get_model()
with tf.Session() as sess:

    # #########################################################
    # feed构造 + 目标处理
    # #########################################################
    infeed = {}

    # 给encoder喂数据
    for key, value in zip(encoder_inputs, samples_encoder_inputs):
        infeed[key.name] = value

    # 给decoder喂数据
    for key, value in zip(decoder_inputs, samples_decoder_inputs):
        infeed[key.name] = value
    infeed[decoder_inputs[output_seq_len].name] = np.zeros([len(train_set)],  dtype=np.int32)
    # 给target_weights喂数据
    for key, value in zip(target_weights, samples_target_weights):
        infeed[key.name] = value

    targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]

    # ###############################
    # 训练阶段 + 评估损失
    # ###############################
    sess.run(tf.global_variables_initializer())
    out, h = sess.run([outputs, hidden], infeed)
    cost = tf.contrib.legacy_seq2seq.sequence_loss(out, targets, target_weights)

    lost = sess.run(cost, infeed)
    print('lost = ', lost)