# 这个模块主要是对网络读入之前做检测，主要检测2个问题：
# 1.节点是否从 0 开始
# 2.是否是有向网络(去重)
# 最后根据检测结果提供读取文件的统一接口

import numpy as np

def standard(item):
    if item[0] < item[1]:
        return item
    else:
        return [item[1],item[0]]

# 读取文件部分
def readFile(file_path):
    f = np.loadtxt(file_path,dtype=int,usecols=(0,1))
    f = np.array(list(map(standard,f)))
    remove_dup = np.unique(f, axis=0)
    start_v = np.min(remove_dup)
    remove_dup = remove_dup - start_v
    edge = np.ndarray.tolist(remove_dup)
    edges = [[str(item[0]), str(item[1])] for item in edge]
    return edges
