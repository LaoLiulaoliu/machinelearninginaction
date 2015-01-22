#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

import regression
from numpy import *
xArr, yArr = regression.loadDataSet('ex0.txt')
ws = regression.standRegres(xArr, yArr)
xMat, yMat = mat(xArr), mat(yArr)
yHat = xMat * ws

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0)
ax.plot(xCopy[:, 1], xCopy*ws)

plt.show()
