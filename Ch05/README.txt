Logistic Regression
1.Cost 函数和 J(θ) 函数是基于最大似然估计推导得到的。
2.最大似然估计求最大值得出θ，于是得出用梯度上升求J(θ)最大值得出θ。
3.用sigmoid和gradient ascent 训练得出最佳系数θ.（有基于向量batch模式，有随机梯度上升模式）
4.根据数据和最佳系数，画出decision boundary，可视化观察正确率。
5.抽出testing样本，测试系数θ带来的正确率，是否需要退回训练阶段，修改步长和迭代次数。
6.给定数据的分类，把数据带入sigmoid计算概率，大于0.5是一种分类，小于等于是另一种。


随机梯度上升算法：占用资源更少，是在线算法，新数据到来就完成参数更新，不需读取整个数据集来批处理运算。
缺失数据：综合sigmoid（0.5）和weight（不更新）更新，决定特征向量取0。这种做法对kNN就不适用。
