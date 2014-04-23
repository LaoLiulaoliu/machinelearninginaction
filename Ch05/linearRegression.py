#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadDataSet(fname='testSet.txt'):
    dataArray = []
    labelArray = []
    with open(fname) as fd:
        for line in fd:
            line = line.strip().split()
            dataArray.append( [1.0, float(line[0]), float(line[1])] )
            labelArray.append( float(line[-1]) )
    return mat(dataArray), mat(labelArray).T

def featureScaling(dataArray):
    X = mat(dataArray)
    X_mean = mean(X, axis=0)
    X_std = std(X, axis=0)
    X_std[0, 0] = 1.0

    m, n = shape(X)
    normData = X - tile(X_mean, (m, 1))
    normData = normData / tile(X_std, (m, 1))
    return normData, X_mean, X_std

def costFunction(X, Y, theta):
    m, n = shape(X)
    cost = X * theta - Y # cost.shape = (m, 1)
    J = 0.5 * m * cost.T * cost
    return J

def gradientDescent(X, Y, theta, alpha, iters):
    """ gradient descent to get local minimum of cost function

    :param theta: theta is an array
    :param iters: iters times
    """
    m, n = shape(X)
    costHistory = zeros((iters, 1))

    for i in xrange(iters):
        theta = theta - alpha * X.T * (X * theta - Y) 
        costHistory[i] = costFunction(X, Y, theta)
    return theta, costHistory

def learningRate(costHistory):
    length = len(costHistory)
    plt.figure()
    plt.subplot(111)
    plt.plot(range(length), costHistory, 'r-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost value')
    plt.title('Different alpha have different learning Rate')
    plt.show()

def plotSurf(X, Y):
    theta0_vals = linspace(-10, 10, 100)
    theta1_vals = linspace(-1, 4, 100)
    J_vals = zeros((len(theta0_vals), len(theta1_vals)))
    for i in xrange(len(theta0_vals)):
        for j in xrange(len(theta1_vals)):
            theta = mat([1, theta0_vals[i], theta1_vals[j]]).T
            J_vals[i, j] = costFunction(X, Y, theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    theta0_vals, theta1_vals = meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    plt.show()


def normalEquation(X, Y):
    """ shape(X) -> (m, n)
        shape(Y) -> (m, 1)

    [(n, m) * (m, n)]' * (n, m) * (m, 1) -> (n, 1)
    """
    theta = (X.T * X)
    theta = theta.I # numpy.linalg.linalg.LinAlgError: Singular matrix
    theta = theta * X.T * Y
    return theta

if __name__ == '__main__':
    data, label = loadDataSet()
    normData, dataMean, dataStd = featureScaling(data)
    plotSurf(normData, label)

    theta0 = zeros((shape(normData)[1], 1))
    alpha = 0.005
    iters = 100
    theta, costHistory = gradientDescent(normData, label, theta0, alpha, iters)
    learningRate( costHistory )

#    theta_n = normalEquation(normData, label)
