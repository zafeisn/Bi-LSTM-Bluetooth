## 进一步处理
鉴于还存在脏数据，于是使用特征重要性对数据进一步处理\
1、首先，使用随机森林方法处理得到的重要性排序：N6、N\
2、随后，分别对每个坐标位置的N6锚点进行数据处理，以此类推其他锚点的数据处理，具体处理办法如下：\
2.1、取N6列的数据最大值，然后再进行环境因子分析，进一步得到在此最大的可接受的信号波动值的范围\
2.2、计算出信号偏移量之后，再进行与最大值相加得到在此位置上一个合适的最小信号强度值\
2.3、最后进行数据处理，小于这个最小值的数据都判断为脏数据即受环境影响较大的数据
## 数据集
* 原数据集：BaseFingerPrintData （4230）
* 预处理后的数据集：select （3568）
## 对比实验
### 一、LSTM
1、原始数据
* 添加网络层\
效果不好，欠拟合\
出现的原因可能是网络过于复杂，需要简化网络层并相应调整batch_size大小

* 调大batch_size\
比之前好一点，但效果不是很明显，并且出现过拟合\
出现的原因可能是数据集不够，需要进行调参（添加dropout层等）或者增大数据集

* 添加dropout层\
在batch_size=64的基础上，添加dropout(0.05)层，效果比之前好一点，但还是存在过拟合现象

2、预处理后的数据
* 网络单元\
416,352,320,480,32,160相对于128,128,128,50,50,128的x坐标的精度更优，但是y坐标精度被损失，同时两种都会出现过拟合现象，前者趋于平缓可能还存在mse下降的趋势，证明调整网络单元能够起到一定的缓解作用

* 学习速率
使用默认lr=0.001反而在x轴上表现更优

* 自动调参\
精度上不一定是最好的，但拟合曲线是最完美的


