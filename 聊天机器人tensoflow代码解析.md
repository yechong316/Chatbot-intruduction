# 聊天机器人tensoflow代码解析

## 开发环境：

tf: 1.10 ，python:3.5

## 案例

```
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
  
outputs, hidden = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    num_encoder_symbols=num_encoder_symbols,
                    num_decoder_symbols=num_decoder_symbols,
                    embedding_size=size,
                    output_projection=None,
                    feed_previous=False,
```

train_set[0] = 2
train_set[1] = 2
train_set[2] = 2
encoder_inputs[0] = (3,)
encoder_inputs[1] = (3,)
encoder_inputs[2] = (3,)
encoder_inputs[3] = (3,)
encoder_inputs[4] = (3,)
decoder_inputs[0] = (3,)
decoder_inputs[1] = (3,)
decoder_inputs[2] = (3,)
decoder_inputs[3] = (3,)
decoder_inputs[4] = (3,)
out[0].shape = (3, 10)
out[1].shape = (3, 10)
out[2].shape = (3, 10)
out[3].shape = (3, 10)
out[4].shape = (3, 10)
h[0].shape = (3, 32)
h[1].shape = (3, 32)



