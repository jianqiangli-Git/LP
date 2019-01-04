# 这个模块主要是对网络读入之前做检测，主要检测2个问题：
# 1.节点是否从 0 开始
# 2.是否是有向网络

import process.linnkprediction as LP
from copy import deepcopy

# 传入边集得到节点的起始序号
def StartVertice(edges):
    v = LP.getVertice(edges)
    v = sorted([int(i) for i in v])
    return v[0]

#决定舍去，没有必要检测,不管是有向还是无向图都可以直接转为无向图
# def Direct(edges):
#     pass