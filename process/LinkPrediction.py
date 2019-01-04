import os
import networkx as nx
import math
import random
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
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

#注意这里因为除数不能为 0 的原因对某对节点的 Hn 指数为 0 的直接做了忽略处理，有些不妥，注意以后作出改进
@not_implemented_for('multigraph')
def hn_resource_allocation_index(G,Hn,ebunch=None):
    def predict(u, v):
        return sum(1 / Hn[w] for w in nx.common_neighbors(G, u, v) if Hn[w]!=0)
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('multigraph')
def adamic_adar_index(G, ebunch=None):
    def predict(u, v):
        return sum(1 / math.log(G.degree(w)) for w in nx.common_neighbors(G, u, v))
    return _apply_prediction(G, predict, ebunch)

#注意这里因为除数不能为 0 的原因对某对节点的 Hn 指数为1 的直接做了忽略处理，有些不妥，注意以后作出改进
@not_implemented_for('multigraph')
def hn_adamic_adar_index(G,Hn, ebunch=None):
    def predict(u, v):
            return sum(1 / math.log(Hn[w]) for w in nx.common_neighbors(G, u, v) if Hn[w]!=1)
    return _apply_prediction(G, predict, ebunch)

def _community(G, u, community):
    """Get the community of the given node."""
    node_u = G.nodes[u]
    try:
        return node_u[community]
    except KeyError:
        raise nx.NetworkXAlgorithmError('No community information')


#字典形式存储每个节点的邻接节点(包括孤立节点)
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

#传入边集得到节点的集合set
def getVertice(edges):
    vertice = []
    for v in edges:
        vertice = vertice+v
    return set(vertice)

#传入图的节点和字典形式的图结构，创建图的邻接矩阵
def createAdj(vertice,cur):
    len_vertice = len(vertice)
    init = np.zeros((len_vertice,len_vertice))
    for v in cur:
        for adj in cur[v]:
            init[int(v)][int(adj)] = 1
    return init

#字典形式存储未连接边的集合(所有节点)
def getNoLinked(net,total_node):
    L = [str(x) for x in range(total_node)]
    noLinked_set = {}
    for i in net.keys():
        dup_L = deepcopy(L)
        dup_L.remove(i)
        noLinked_list = filter(lambda n:n not in net[i],dup_L)
        newList = list(noLinked_list)
        noLinked_set[i] = newList
    return noLinked_set

#传入字典形式的图结构，得到字典形式的每个节点的度
def getDictDegree(G):
    degree = nx.degree(G)
    return degree

#传入所有节点集合vertice，得到每个节点跟需要预测的节点之间的共同邻居有哪些
def getCommonNeighbor_node(G,cur,vertice):
    commom = {}
    set_ver = set(vertice)
    for v in vertice:
        notDirecLinked = set_ver-set(cur[v])
        for n in notDirecLinked:
            if int(v)<int(n):
                commom[(v,n)] = sorted(nx.common_neighbors(G,n,v))
    return commom

#把边集分成测试集和训练集，列表形式,传入的比率是测试集的占比
def devide_train_and_test_set(edges,ratio):
    dup_edges = deepcopy(edges)
    test_set_num = int(len(edges)*ratio)
    test_set = random.sample(edges,test_set_num)
    train_set = list(filter(lambda n: n not in test_set, dup_edges))
    return train_set,test_set

#输入当前图的字典形式和所有的节点，得到需要预测的边{点:{需要预测的边}}
def getEdgeForPredict(cur,vertice):
    edgeForPredict = {}
    set_ver = set(vertice)
    for v in vertice:
        ver = deepcopy(set_ver)
        ver.discard(v)
        notDireclinked = ver-set(cur[v])
        edgeForPredict[v] = notDireclinked
    return edgeForPredict

#得到图的核数字典
def getCore(G):
    core_dict = nx.core_number(G)
    return core_dict

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

#cur 是字典形式存储的当前图的结构， degreeDict 是字典形式存储当前图每个节点的度
def getHIndice(cur,degreeDict,c):
    HIndice = {}
    order = 0
    HIndice["h(%s)"%order]=degreeDict
    indice = []
    for order in range(1,20):
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

#传入图的字典形式和度的字典，得到这个图的概率转移矩阵 PMatrix
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

#传入n-step转移概率矩阵得到每个节点转移到其他节点的概率矩阵
def getVerticeTransformProbMatrix(NPmatrix):
    n = len(NPmatrix[0])
    ver_matrix = np.eye(n)
    prob_matrix = np.dot(ver_matrix, NPmatrix)
    return prob_matrix

def CN_num(adjMatrix,vertice,cur):
    CN_info = {}
    set_ver = set(vertice)
    square = adjMatrix.dot(adjMatrix)
    print(square)
    for v in vertice:
        notDirecLinked = set_ver-set(cur[v])
        for n in notDirecLinked:
            if int(n)>int(v):
                CN_info[(v,n)]=square[int(v)][int(n)]
    return CN_info

#通过调用networkX库函数得到AA_RA指标，并存储到AA_info，RA_info字典中
def AA_RA(G):
    AA_info={}
    RA_info={}
    AA=nx.adamic_adar_index(G)
    for n,v,p in AA:
        if int(n)>int(v):
            AA_info[(v,n)]=p
        else:
            AA_info[(n,v)]=p
    RA=nx.resource_allocation_index(G)
    for i,j,k in RA:
        if int(i)>int(j):
            RA_info[(j,i)]=k
        else:
            RA_info[(i,j)]=k
    return AA_info,RA_info

#通过修改库函数的AA_RA函数，假如Hn得到Hn各个指数的Hn_AA_RA指标，存到Hn_AA_info和Hn_RA_info字典中
def getHn_AA_RA(Hn):
    Hn_AA_info = {}
    Hn_RA_info = {}
    Hn_AA = {}
    Hn_RA = {}
    for i in Hn:
        pred_AA = hn_adamic_adar_index(G, Hn[i])
        pred_RA = hn_resource_allocation_index(G, Hn[i])
        for u, v, p in pred_AA:
            if int(u)<int(v):
                Hn_AA[(u,v)] = p
            else:
                Hn_AA[(v, u)] = p
        for s, j, k in pred_RA:
            if int(s) < int(j):
                Hn_RA[(s, j)] = k
            else:
                Hn_RA[(j, s)] = k
        Hn_RA_info[i] = Hn_RA
        Hn_AA_info[i] = Hn_AA
    return Hn_AA_info,Hn_RA_info

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

#传入需要预测的边字典,Hn,边数,每个节点到其他节点的转移概率矩阵得到使用Hn各个中间过程指标的待预测边的t步LRW分,其中Hn代表一系列的H指数h(0~n)
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

#传入步数t，得到2~t步(含t步)概率转移矩阵字典{"步数t":t步概率转移矩阵}，其中1步转移矩阵就是概率转移矩阵P
def getNPMatrixDict(P,t):
    NPmatrixDict = {}
    NP = getNPmatrix(P, 2)
    NPmatrixDict["2"] = NP
    for i in range(3, t + 1):
        NP = np.dot(NP, NP)
        NPmatrixDict[str(i)] = NP
    return NPmatrixDict

#传入n-step转移概率矩阵得到每个节点转移到其他节点的概率矩阵
def getVerTransProbMatrix(NPmatrix):
    n = len(NPmatrix[0])
    ver_matrix = np.eye(n)
    prob_matrix = np.dot(ver_matrix, NPmatrix)
    return prob_matrix

#传入2~t步(含t步)概率转移矩阵字典和节点数,得到每步每个节点到其他节点的转移概率矩阵字典{"步数t":节点转移概率矩阵}
def getNVerTransProbMatrixDict(NPmatrixDict,ver_num):
    NProbMatrixDict = {}
    ver_matrix = np.eye(ver_num)
    for n in NPmatrixDict:
        NProbMatrixDict[n] = np.dot(ver_matrix,NPmatrixDict[n])
    return NProbMatrixDict

#传入2~N步概率转移矩阵字典,得到各个待预测边2~t步转移分之和HSRW,其中H代表单纯的H指数
def getHSRW(H,edgeForPrediction,NProbMatrix,edge_num):
    HSRW_info = {}
    for i in edgeForPrediction:
        for j in edgeForPrediction[i]:
            if int(i) < int(j):
                HSRW_info[(i,j)] = 0

    for n in NProbMatrix:
        print("======================{index}=======================".format(index=n))
        for edge in HSRW_info:
            i = list(edge)[0]
            j = list(edge)[1]
            score = (H[i] * NProbMatrix[n][int(i)][int(j)] + H[j] * NProbMatrix[n][int(j)][int(i)]) /(2*edge_num)
            HSRW_info[(i, j)] = HSRW_info[(i, j)] + score
    return HSRW_info

#输入Hn各个中间过程指标,2~N步概率转移矩阵字典,得到使用Hn各个中间过程指标的待预测边的t步HSRW分，步数t蕴含在NPmatrixDict中，由NPmatrixDict得到节点转移概率矩阵NProbMatrix
def getHn_HSRW(Hn,edgeForPrediction,NProbMatrixDict,edge_num):
    Hn_HSRW_info = {}
    Hn_HSRW = {}

    for i in edgeForPrediction:
        for j in edgeForPrediction[i]:
            if int(i) < int(j):
                Hn_HSRW[(i, j)] = 0
    temp_Hn_HSRW = Hn_HSRW

    for k in Hn:
        Hn_HSRW = deepcopy(temp_Hn_HSRW)
        print("===============================Hn_HSRW============================")
        print("================================={k}=============================".format(k=k))
        for n in NProbMatrixDict:
            print("------------{index}-step----------".format(index=n))
            for edge in Hn_HSRW:
                i = list(edge)[0]
                j = list(edge)[1]
                score = (Hn[k][i] * NProbMatrixDict[n][int(i)][int(j)] + Hn[k][j] * NProbMatrixDict[n][int(j)][int(i)]) / (2 * edge_num)
                Hn_HSRW[(i, j)] = Hn_HSRW[(i, j)] + score
        Hn_HSRW_info[k] = Hn_HSRW
    return Hn_HSRW_info

# def Hindice_AA_RA(vertice,cur,adj,Hindice):
#     Hn_AA_info = {}
#     Hn_RA_info = {}
#     AA_info = {}
#     RA_info = {}
#     set_ver = set(vertice)
#     for h in Hindice:
#         for v in vertice:
#             notLinked = set_ver-set(cur[v])
#             for n in notLinked:
#                 if int(n)>int(v):
#                     total_AA = 0
#                     total_RA = 0
#                     dup_vertice = deepcopy(vertice)
#                     dup_vertice.remove(v)
#                     for j in dup_vertice:
#                         if adj[int(v)][int(j)] !=0 and adj[int(j)][int(n)] != 0:
#                             Hn = Hindice[h][j]
#                             if Hn==1:
#                                 pass
#                             else:
#                                 total_AA = total_AA+1/math.log(Hn)
#                                 total_RA = total_RA+1/Hn
#                     AA_info[(v,n)] = total_AA
#                     RA_info[(v,n)] = total_RA
#                     if (v,n) not in set(AA_info.keys()):
#                         AA_info[(v,n)] = 0
#                     if (v, n) not in set(AA_info.keys()):
#                         AA_info[(v, n)] = 0
#         Hn_AA_info[h] = AA_info
#         Hn_RA_info[h] = RA_info
#     return Hn_AA_info,Hn_RA_info

#传入测试集和未连接边的 {预测边:连接概率} 的字典关系，得到AUC
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
    len_noLinked = len(noLinked_set_score_info)
    test_list=sorted(test_set_score_info)
    noLinked_list = sorted(noLinked_set_score_info)
    for t in test_list:
        for n in noLinked_list:
            if t>n:
                auc = auc+1
            elif t==n:
                auc = auc+0.5
            else:
                break
    auc = auc/(len_test*len_noLinked)
    return auc


if __name__ == "__main__":
    Yeast_path = r"C:\Users\Tang\Desktop\data\Yeast.txt"
    edges=read_file(Yeast_path)
    tr, te = devide_train_and_test_set(edges, 0.1)
    print("te:", te)
    # G.add_edges_from(tr)
    # nodes = set(G.nodes())
    v = set(getVertice(edges))
    net = dictNet(tr, v)
    G = nx.Graph(net)
    # adj = createAdj(v, net)
    # discard_node = v-nodes
    # print("discard_nodes:", discard_node)
    # G.add_nodes_from(discard_node)
    degree = getDictDegree(G)
    core = getCore(G)
    AA, RA = AA_RA(G)
    print("AUC_AA:",getAUC(te, AA))
    print("AUC_RA:",getAUC(te, RA))
    h = getHIndice(net, degree, core)
    print("HnIndex", h)
    # H_AA, H_RA = Hindice_AA_RA(v, net, adj, h)
    # H_AA,H_RA = getHn_AA_RA(h)
    # for i in H_AA:
    #     print(i)
    #     auc_AA = getAUC(te,H_AA[i])
    #     print("AUC_AA_{index}:".format(index=i), auc_AA)
    #     print("")
    # for j in H_RA:
    #     print(j)
    #     auc_RA = getAUC(te, H_RA[j])
    #     print("AUC_RA_{index}:".format(index=j), auc_RA)
    #     print("")
    # v= getVertice(edges)
    eforPre = getEdgeForPredict(net,v)
    P=creatPMatrix(net,degree)
    # NP = getNPmatrix(P,3)
    # probMatrix = getVerTransProbMatrix(NP)
    # lrw = LRW(eforPre,degree,len(tr),probMatrix)
    # print("AUC_LRW:",getAUC(te,lrw))
    # Hn_LRW = getHn_LRW(eforPre,h,len(tr),probMatrix)
    # for k in Hn_LRW:
    #     print(k)
    #     auc_LRW = getAUC(te, Hn_LRW[k])
    #     print("AUC_LRW_{index}:".format(index=k), auc_LRW)
    #     print("")

    print("=================================Calculating HSRW=========================================")
    Hn_Pmatrix = getNPMatrixDict(P,5)
    NProbMatrix = getNVerTransProbMatrixDict(Hn_Pmatrix, len(v))
    HSRW = getHSRW(h['h(1)'], eforPre, NProbMatrix, 5)
    auc_HSRW = getAUC(te, HSRW)
    print("auc_HSRW:", auc_HSRW)
    print("=================================Calculating Hn_HSRW======================================")
    Hn_HSRW = getHn_HSRW(h, eforPre, NProbMatrix, len(v))
    for n in Hn_HSRW:
        print("auc_Hn_HSRW_{n}".format(n=n), ":", getAUC(te, Hn_HSRW[n]))

        # plt.figure()
    # nx.draw(G,with_labels=True)
    # plt.show()

