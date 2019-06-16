# 聊天机器人
可以与机器人进行简短的聊天对话。
# 开发环境
TensorFlow：1.10.0(后续更新Pytorch)
Python：3.5.2
jieba:0.39
# 使用办法

运行main.py，详见下文：

```python  
  
总共有三种模式：
0 : 训练模式
1 : BLEU模式
2 : 测试模式。
mode = 2
work_mode(mode)
```
# 待完善的地方
提问来自语料库中的句子时，准确度非常高，目前还有很多瑕疵，后续不断完善中，代码将逐步更新创建过程
1.回答语句有空格，
2.未对语料库分桶
3.暂未做FLASKweb处理
4.后续更新基于Pytorch建立聊天机器人
5.使用更多类型的语料库进行更新
6.暂未重构代码，使之可以方便商用。
# 计算时间
 本人电脑为GTX1060 6G显卡，计算时间大概在10min左右，迭代到1000+次后，损失值徘徊在0.088左右，开始进行测试，见下图
![img1](https://github.com/yechong316/Chatbot-intruduction/blob/master/result/%E8%81%8A%E5%A4%A91.png)
![img1](https://github.com/yechong316/Chatbot-intruduction/blob/master/result/%E8%81%8A%E5%A4%A92.png)


# 致谢
数据来源：语料下载至  [here](https://github.com/codemayq/chinese_chatbot_corpus)，非常感谢作者codemayq的无私奉献，已星标follow。
部分代码：李闯 warmheartli whlichuang@126.com，同样已星标加follow~~~
# 作者
本人两个名字 叶重（yechong316）/Jaime Lannister 感兴趣的朋友请给个星星~~~
