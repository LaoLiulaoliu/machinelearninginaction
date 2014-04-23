#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miracle (at) gmail.com>

import trees
import treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
fr.close()
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
treePlotter.createPlot(lensesTree)
