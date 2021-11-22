# 基于bert4keras的CLUE基准代码
真·“Deep Learning for Humans”

## 简介
- 博客：https://kexue.fm/archives/8739

（说实话我也不知道要补充点啥，我觉得代码本身够清晰了，如果还有什么疑问，欢迎提issue～）

## 使用
模型和优化器定义在`snippets.py`里边，如果要更换模型，修改`snippets.py`即可。

优化器使用了AdaFactor，这是因为它对参数范围具有较好的适应性，是一个较优（但不一定是最好）的默认选择。

## 环境
- 软件：bert4keras>=0.10.8
- 硬件：博客中的成绩是用一张Titan RTX（24G）跑出来的，如果你显存不够，可以适当降低batch_size，并启用梯度累积。

## 其他
- 英文GLUE榜单的bert4kears基准：https://github.com/nishiwen1214/GLUE-bert4keras

## 交流
QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn
