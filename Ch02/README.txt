The code for the examples in Ch.2 is contained in the python module: kNN.py.
The examples assume that datingTestSet.txt is in the current working directory.  
Folders testDigits, and trainingDigits are assumed to be in this folder also.  

k nearest neighbor.
1. 处理数据，从数据库读入内存；
2. 基础数据做normalization，标签数据；
3. 给一个数据和k，根据欧式距离，计算出给定数据最近的k条记录；
4. 利用现有数据，计算验证kNN算法的在这些数据上的正确率。
