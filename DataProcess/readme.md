本数据为2021年12月28日于机电大楼A631实验室采集的蓝牙信号强度指示数据，实验室内两侧各对称布置3个信标，同时包括前后各1个信标，总共8个蓝牙信标  
### 以下为信标参数
**信标名称：** N1 N2 N3 N4 N5 N6 N7 N8  
**芯片型号：** nRF52810  
**发射功率：** 6dBm（0-6可调）  
**电池供电：** CR2032纽扣电池、3V电压、100%电量  
**其他功能：** 可控制电池开关  

### 以下为设备参数 
**移动设备：** 坚果Pro2  
**物理地址：** B4:0B:44:F3:74:66  
**系统版本：** Android 7.1.1  
**采集时间：** 2021年12月28日  -  2021年12月31日（补充）

### 以下为数据参数
**位置坐标：** X、Y、Z；其中Z坐标统一设置为0，考虑到实际应用中是三维空间，故字段保留，后续会是研究的延伸和拓展  

**信号强度：** N1 N2 N3 N4 N5 N6 N7 N8 W1 W2 W3 W4 W5 W6 W7 W8 W9 W0 F1 F2；其中，N表示内，W表示外，F表示空闲，但目前后12个信标未使用，后续考虑扩大定位空间的范围再进行布置  

**电量参数：** Electricity-信标电池电量；设计的时候考虑每个电池用电可能不一致，故将每个信标的电量进行字符串的拼接，以每3位为一个电池电量信息，用的时候只需要按需取即可  

**其他参数：** Start_Time-扫描时间（只记录开始的扫描时间）、Phone_MAC-设备物理地址、Phone_Brand-设备型号、Phone_Android-设备系统信息、Scan_Duration-扫描时长、Scan_Interval-设备扫描功率、Tx_Power-信标发射功率；其中，后2个数据需要Android10（也就是SDK的版本需要30）以上才能使用，目前暂定设置为0  

**总记录数：** 1693条  

**数据名称：** 训练集  

**数据用途：** 作为预处理的第一部分的数据集，当完成基本指纹库的创建之后，以便后续使用监督算法进一步的建立完整的指纹数据库   

### 以下为算法介绍
**核心步骤：**   
<font color=Red>
① 采用KMeans聚类（取K=2）进行非人工数据挑选，降低人工的时间成本，以建立基本数据库  
② 根据得到的基本数据库，对其使用KMeans聚类（手肘法）进行人工数据筛选，这部分比较费时费力，但由于基本数据库的数据量较少，相比之下偏优；其中，聚类次数与参考点的个数有关，本数据为61个参考点，故需要进行61次聚类和61次人工筛选，以建立基本指纹库  
③ 完成基本指纹库的建立之后，开始在后续更大的数据集或扩充数据集上进行监督聚类，该操作不需要人工参与，因此在很大程度上降低了人工成本  
</font>

**算法比较：**  
<font color=Blue>使用本数据在不考虑处理时间的情况下，比较聚类次数（本算法为：61次；KMeans：61次）这是因为都使用了手肘法进行聚类操作；但由于深度学习技术是基于数据量的一项技术，因此数据往往是需要再进一步扩充的。
  
对于KMeans来说，每扩充一次就需要进行61次聚类，也就是说需要进行61次人工筛选，同时61次也只是在本数据中使用，运用到其他更多参考点的数据集显然该方法就显得很不适应，同时KMeans对大数据的收敛速度较慢，进一步加大了数据处理的时间复杂度；而本课题所提算法对于后续数据扩充并不需要进行人工加以干预，同时使用半监督算法避免了收敛问题，因此性能更优 </font>

### 以下为具体实现
1、先读取本数据，然后再对坐标X和Y进行字符串的拼接，并加以存入数据中作为一组标签   
2、选择特征列和标签列，使用字典将特征和每个标签进行对应，以便后续使用聚类操作  
3、创建聚类函数，从上述字典中分别取出特征和标签对进行依次聚类，并将聚类结果挨个保存（目前只做到分别存储，后续考虑向尾部添加）  
4、根据文件中的异常列进行取舍，并进行基本数据库的建立  
5、完成基本数据库的创建之后，再对其进一步的聚类，这部分使用手肘法，画图选择合适的聚类数进行聚类，最后进行人工挑选，建立基本指纹库  
6、完成基本指纹库的创建之后，开始对扩充数据库进行以半监督算法的应用，其中需要手动添加部分异常数据，以增大算法的容错率，这部分需要人工经验操作，是值得后续进一步研究的内容  
7、半监督算法采用LabelSpreading，其中train的真实标签为基本指纹库和人工掺杂的标签，预测标签预先统一设置为-1，使用算法进行聚类预测  