# coding:utf-8

import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq


def get_model():
    """构造模型
    """

    learning_rate = tf.Variable(float(init_learning_rate), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in range(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in range(output_seq_len + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
    for i in range(output_seq_len):
        target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

    # decoder_inputs左移一个时序作为targets
    targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]

    # cell = tf.contrib.rnn.BasicLSTMCell(size)
    dropout = 1
    num_layers = 3
    cell = tf.contrib.rnn.BasicLSTMCell(size)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)  # 纵向上有两个LSTM

    # 这里输出的状态我们不需要
    outputs, _ = seq2seq.embedding_attention_seq2seq(
                        encoder_inputs,
                        decoder_inputs[:output_seq_len],
                        cell,
                        num_encoder_symbols=num_encoder_symbols,
                        num_decoder_symbols=num_decoder_symbols,
                        embedding_size=size,
                        output_projection=None,
                        feed_previous=feed_previous,
                        dtype=tf.float32)

    # 计算加权交叉熵损失
    loss = seq2seq.sequence_loss(outputs, targets, target_weights)
    # 梯度下降优化器
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # 优化目标：让loss最小化
    # update = opt.apply_gradients(opt.compute_gradients(loss))
    # 模型持久化
    saver = tf.train.Saver(tf.global_variables())

    return encoder_inputs, decoder_inputs, target_weights, outputs, loss, opt, saver, learning_rate_decay_op, learning_rate