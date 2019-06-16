# 聊天机器人

# 开发环境

TensorFlow：1.10.0

Python：3.5.2

jieba:0.39

# 使用办法

运行main.py，详见下文：

··```<<<<<<<<<<<

```
from demo import work_mode



'''
总共有三种模式：
0 : 训练模式
1 : BLEU模式
2 : 测试模式，注：测试模式暂不开放。
'''

mode = 2
work_mode(mode)
```

# 致谢

语料下载于  [here](https://github.com/codemayq)<https://github.com/codemayq/chinese_chatbot_corpus>)，对此表示对作者感谢，并以星标作者。

代码部分：致谢



# 计算时间

本人电脑为GTX1060 6G显卡，计算时间大概在10min左右，迭代到1000+次后，损失值徘徊在0.088左右，开始进行测试，见下图

![](D:\自然语言处理实战\聊天机器人\Chatbot-intruduction\result\聊天1.png)

![](D:\自然语言处理实战\聊天机器人\Chatbot-intruduction\result\聊天2.png)

可以看到，当采用语料库中的句子进行提问，回答质量非常高，尤其是在问到“什么是ai”和“什么是蜘蛛侠”，回答结果无比正确，受限语料库句子数量较小，如果问到非语料库中的句子，回答质量就比较差。