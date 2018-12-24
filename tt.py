import random
from math import log
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ['resource_allocation_index',
           'hn_resource_allocation_index',
           'jaccard_coefficient',
           'adamic_adar_index',
           'hn_adamic_adar_index',
           'preferential_attachment',
           'cn_soundarajan_hopcroft',
           'ra_index_soundarajan_hopcroft',
           'within_inter_cluster']


def _apply_prediction(G, func, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)
    return ((u, v, func(u, v)) for u, v in ebunch)

@not_implemented_for('multigraph')
def resource_allocation_index(G, ebunch=None):
    def predict(u, v):
        return sum(1 / G.degree(w) for w in nx.common_neighbors(G, u, v))
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('multigraph')
def hn_resource_allocation_index(G,Hn,ebunch=None):
    def predict(u, v):
        return sum(1 / Hn[w] for w in nx.common_neighbors(G, u, v) if Hn[w]!=0)
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('multigraph')
def adamic_adar_index(G, ebunch=None):
    def predict(u, v):
        return sum(1 / log(G.degree(w)) for w in nx.common_neighbors(G, u, v))
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('multigraph')
def hn_adamic_adar_index(G,Hn, ebunch=None):
    def predict(u, v):
            return sum(1 / log(Hn[w]) for w in nx.common_neighbors(G, u, v) if Hn[w]!=1)
    return _apply_prediction(G, predict, ebunch)

def _community(G, u, community):
    """Get the community of the given node."""
    node_u = G.nodes[u]
    try:
        return node_u[community]
    except KeyError:
        raise nx.NetworkXAlgorithmError('No community information')

def getHIndice(cur,degreeDict,c):
    HIndice = {}
    order = 0
    HIndice["h(%s)"%order]=degreeDict
    indice = []
    for order in range(1,10):
        Hn = {}
        for i in cur:
            for j in cur[i]:
                indice.append(HIndice["h({a})".format(a=order-1)][j])
            h = H(indice)
            Hn[i] = h
            indice.clear()
        #print(Hn)
        HIndice["h({b})".format(b=order)]=Hn
        if Hn==c:
            break
    return HIndice

def getDegree(G):
    degree = nx.degree(G)
    return degree

def creatPMatrix(net,degreeDict):
    n = len(net.keys())
    PMatrix = np.zeros((n,n))
    for i in net.keys():
        if len(net[i])==0:
            continue
        for j in net[i]:
            PMatrix[int(i)][int(j)] = 1/degreeDict[i]
    return PMatrix

#传入概率转移矩阵PMatrix，得到n步概率转移矩阵
def getNPmatrix(PMatrix,n):
    for i in range(n-1):
        PMatrix = np.dot(PMatrix,PMatrix)
    return PMatrix

def getEdgeForPredict(cur,vertice):
    edgeForPredict = {}
    set_ver = set(vertice)
    for v in vertice:
        ver = deepcopy(set_ver)
        ver.remove(v)
        notDireclinked = ver-set(cur[v])
        edgeForPredict[v] = notDireclinked
    print("forPrediction")
    print(edgeForPredict)
    return edgeForPredict

def devide_train_and_test_set(edges,ratio):
    dup_edges = deepcopy(edges)
    test_set_num = int(len(edges)*ratio)
    test_set = random.sample(edges,test_set_num)
    train_set = list(filter(lambda n: n not in test_set, dup_edges))
    return train_set,test_set

# 传入一个节点的邻居的度，得到这个节点的 h-indice
def H(real):
    h=0
    r = sorted(real,reverse=True)
    for i in range(1,len(real)+1):
        if i>r[i-1]:
            h=i-1
            break
        if i==r[i-1] or i==len(real):
            h=i
            break
    return h

def dictNet(edges,ver):
    net = {}
    for edge in edges:
        if edge[0] not in net.keys():
            net.setdefault(edge[0],[]).append(edge[1])
        else:
            net[edge[0]].append(edge[1])
        if edge[1] not in net.keys():
            net.setdefault(edge[1], []).append(edge[0])
        else:
            net[edge[1]].append(edge[0])
    vertice=set(net.keys())
    for s in (ver-vertice):
        net[s]=[]
    return net

def getCore(G):
    core_dict = nx.core_number(G)
    return core_dict

def getHn_AA_RA(Hn):
    Hn_AA_info = {}
    Hn_RA_info = {}
    Hn_AA = {}
    Hn_RA = {}
    for i in Hn:
        print("Get to calculate {i}_AA_RA".format(i=i))
        pred_AA = hn_adamic_adar_index(G, Hn[i])
        pred_RA = hn_resource_allocation_index(G, Hn[i])
        for u, v, p in pred_AA:
            if int(u)<int(v):
                Hn_AA[(u,v)] = p
            else:
                Hn_AA[(v, u)] = p
        for a, j, k in pred_RA:
            if int(a) < int(j):
                Hn_RA[(a, j)] = k
            else:
                Hn_RA[(j, a)] = k
        Hn_RA_info["{index}".format(index=i)] = Hn_RA
        Hn_AA_info["{index}".format(index=i)] = Hn_AA
    return Hn_AA_info,Hn_RA_info

#传入n-step转移概率矩阵得到每个节点转移到其他节点的概率矩阵
def getVerticeTransformProbMatrix(NPmatrix):
    n = len(NPmatrix[0])
    ver_matrix = np.eye(n)
    prob_matrix = np.dot(ver_matrix, NPmatrix)
    return prob_matrix

#传入n步转移矩阵，待预测的边的字典，图节点的度字典，边的总数得到每对待预测边的lrw分
def LRW(edgeForPrediction,degreeDict,edge_num,prob_matrix):
    lrw_info = {}
    print("probMatrix:")
    print(prob_matrix)
    for i in edgeForPrediction:
        for j in edgeForPrediction[i]:
            if int(i)<int(j):
                lrw_info[(i,j)] = (degreeDict[i]*prob_matrix[int(i)][int(j)]+degreeDict[j]*prob_matrix[int(j)][int(i)])/(2*edge_num)
    return lrw_info

#将Hn的各个中间指标应用到LRW中，得到Hn各个中间指标的lrw分，其中Hn代表一系列的H指数h(0~n)
def getHn_LRW(edgeForPrediction,Hn,edge_num,prob_matrix):
    Hn_LRW_info = {}
    for n in Hn:
        Hn_LRW = {}
        for i in edgeForPrediction:
            for j in edgeForPrediction[i]:
                if int(i) < int(j):
                    Hn_LRW[(i, j)] = (Hn[n][i] * prob_matrix[int(i)][int(j)] + Hn[n][j] * prob_matrix[int(j)][int(i)]) / (2 * edge_num)
        Hn_LRW_info[n] = Hn_LRW
    return Hn_LRW_info

#传入概率转移矩阵P和转移步数t得到各个待预测边2~t步转移分之和,其中H代表单纯的H指数
def getHSRW(P,H,edgeForPrediction,prob_matrix,edge_num,t):
    NPmatrix = {}
    HSRW_info = {}
    NP = getNPmatrix(P, 2)
    NPmatrix["2"] = NP
    for i in range(3,t+1):
        NP = np.dot(NP,NP)
        NPmatrix[str(i)] = NP

    for i in edgeForPrediction:
        for j in edgeForPrediction[i]:
            if int(i) < int(j):
                HSRW_info[(i,j)] = 0

    for n in NPmatrix:
        print("======================{index}=======================".format(index=n))
        for edge in HSRW_info:
            i = list(edge)[0]
            j = list(edge)[1]
            score = (H[i] * NPmatrix[n][int(i)][int(j)] + H[j] * NPmatrix[n][int(j)][int(i)]) /(2*edge_num)
            HSRW_info[(i, j)] = HSRW_info[(i, j)] + score
    return HSRW_info


#传入列表形式的测试集，字典形式的分数信息，输出auc
def getAUC(test_set,info):
    auc = 0
    test_set_score_info = []
    noLinked_set_score_info = []
    for i in set(info.keys()):
        if list(i) in test_set:
            test_set_score_info.append(info[i])
        else:
            noLinked_set_score_info.append(info[i])
    len_test = len(test_set_score_info)
    print("len_test------>",len_test)
    len_noLinked = len(noLinked_set_score_info)
    print("len_nolinked-------->",len_noLinked)
    print("info------>",len(info.keys()))
    test_list=sorted(test_set_score_info)
    noLinked_list = sorted(noLinked_set_score_info)
    for t in test_list:
        for n in noLinked_list:
            if t>n:
                auc = auc+0.5
            elif t==n:
                auc = auc+1
            else:
                break

    auc = auc/(len_test*len_noLinked)
    return auc

edges=[["0","1"],["0","2"],["1","3"],["1","4"],["1","5"],["2","5"]]
v=set(["0","1","2","3","4","5"])
tr,te=devide_train_and_test_set(edges,0.4)
print("tr:",tr)
net = dictNet(tr,v)
print("net:",net)
G = nx.Graph(net)
c=getCore(G)
d = getDegree(G)
print("degree:",d)
P=creatPMatrix(net,d)
print("Pmatrix:")
print(P)
P2=getNPmatrix(P,2)
print("P2:")
print(P2)
Hn=getHIndice(net,d,c)
print("Hn:",Hn)
preds = nx.adamic_adar_index(G)
print("--------------AA-------------")
for u, v, p in preds:
    print('(%s, %s) -> %.8f' % (u, v, p))

print("")
print("--------------RA-------------")
pred = nx.resource_allocation_index(G)
for u, v, p in pred:
    print('(%s, %s) -> %.8f' % (u, v, p))
print("")

H_AA,H_RA = getHn_AA_RA(Hn)
print("H_AA:",H_AA)
print("H_RA:",H_RA)
for i in H_AA:
    print(i)
    auc_AA = getAUC(te,H_AA[i])
    print("AUC_AA_{index}:".format(index=i), auc_AA)
    print("")

for j in H_RA:
    print(j)
    auc_RA = getAUC(te, H_RA[j])
    print("AUC_RA_{index}:".format(index=j), auc_RA)
    print("")
v=["0","1","2","3","4","5"]
eforPre = getEdgeForPredict(net,v)
probMatrix=getVerticeTransformProbMatrix(P2)
lrw=LRW(eforPre,d,4,probMatrix)
print("lrw:",lrw)
getHn_LRW(eforPre,Hn,4,probMatrix)




getHSRW(P,Hn['h(1)'],eforPre,probMatrix,4,4)
plt.figure()
nx.draw(G,with_labels=True)
plt.show()