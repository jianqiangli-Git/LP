from copy import deepcopy
import random
import numpy as np
import math
import time
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
        return sum(1 / Hn[w] for w in nx.common_neighbors(G, u, v))
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('multigraph')
def adamic_adar_index(G, ebunch=None):
    def predict(u, v):
        return sum(1 / math.log(G.degree(w)) for w in nx.common_neighbors(G, u, v))
    return _apply_prediction(G, predict, ebunch)

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

#edges=[("0","1"),["1","2"],["1","3"],["1","4"],["1","5"],["2","4"],["2","3"],["2","5"],["3","4"],["3","6"],["3","8"],["6","7"],["7","8"],["5","9"],["9","10",5],["9","11",9],["9","12",3],["9","13",3],["9","14",5]]
# dic1 = {("0","1"):1,("1","2"):2,("2","3"):1,("3","4"):4,("2","4"):4,("1","4"):5}
# dic2 = {"1":1,"3":3,"5":5}
edges=[["0","1"],["0","2"],["1","3"],["1","4"],["1","5"],["2","5"]]
#edges=[(0,1,2),(1,2,1),(1,3,2)]

def read_file(file_path):
    edges = []
    with open(file_path, 'r') as f:
        edges_num = 0
        while True:
            edge = f.readline()
            if not edge:
                print("total eges:", edges_num)
                break
            else:
                edges_num = edges_num + 1
                edge = edge.strip().split()
                if int(edge[0]) > int(edge[1]):
                    edges.append(list((edge[1],edge[0])))
                else:
                    edges.append(edge)
    return edges

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


def getNoLinked(net,total_node):
    L = [str(x) for x in range(total_node)]
    noLinked_set = {}
    for i in net.keys():
        dup_L = deepcopy(L)
        dup_L.remove(i)
        noLinked_list = filter(lambda n:n not in net[i],dup_L)
        newList = list(noLinked_list)
        noLinked_set[i] = newList
    print("noLinked------------->", noLinked_set)

def devide_train_and_test_set(edges,ratio):
    dup_edges = deepcopy(edges)
    test_set_num = int(len(edges)*ratio)
    test_set = random.sample(edges,test_set_num)
    train_set = list(filter(lambda n: n not in test_set, dup_edges))
    return train_set,test_set

def getDegree(G):
    degree = nx.degree(G)
    return degree


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

def getCore(G):
    core_dict = nx.core_number(G)
    return core_dict

#传入测试集和未连接边的 {预测边:连接概率} 的字典关系，得到AUC
def getAUC(test_set,info,name):
    print("--------{name}--------".format(name=name))
    t1=time.time()
    auc = 0
    test_set_score_info = []
    noLinked_set_score_info = []
    for i in info.keys():
        if list(i) in test_set:
            test_set_score_info.append(info[i])
        else:
            noLinked_set_score_info.append(info[i])
    print("noLink_score:",noLinked_set_score_info)
    print("test_score:",test_set_score_info)
    len_test = len(test_set_score_info)
    print("len_test------>",len_test)
    len_noLinked = len(noLinked_set_score_info)
    print("len_nolinked-------->",len_noLinked)
    print("info------>",len(info.keys()))
    test_list=sorted(test_set_score_info)
    noLinked_list = sorted(noLinked_set_score_info)
    for t in test_list:
        for n in noLinked_list:
            if t==n:
                auc = auc+0.5
            elif t>n:
                auc = auc+1
            else:
                break
    auc = auc/(len_test*len_noLinked)
    t2=time.time()
    print("AUC_time:",t2-t1)
    return auc

def createAdj(vertice,cur):
    len_vertice = len(vertice)
    init = np.zeros((len_vertice,len_vertice))
    for v in cur:
        for adj in cur[v]:
            init[int(v)][int(adj)] = 1
    return init

def getVertice(edges):
    vertice = []
    for v in edges:
        vertice = vertice+v
    return set(vertice)

def CN_num(adjMatrix,vertice,cur,linkForPredict):
    CN_info = {}
    set_ver = set(vertice)
    square = adjMatrix.dot(adjMatrix)
    for v in linkForPredict:
        for n in linkForPredict[v]:
            if int(n)>int(v):
                CN_info[(v,n)]=square[int(v)][int(n)]
    return CN_info

def getEdgeForPredict(cur,vertice):
    edgeForPredict = {}
    set_ver = vertice
    for v in vertice:
        ver = deepcopy(set_ver)
        ver.discard(v)
        notDireclinked = ver-set(cur[v])
        edgeForPredict[v] = notDireclinked
    print("forPrediction")
    print(edgeForPredict)
    return edgeForPredict

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

#计算Hn指数的AA和RA
def Hindice_AA_RA(vertice,cur,adj,Hindice):
    Hn_AA_info = {}
    Hn_RA_info = {}
    AA_info = {}
    RA_info = {}
    set_ver = set(vertice)
    for h in Hindice:
        for v in vertice:
            notLinked = set_ver-set(cur[v])
            for n in notLinked:
                if int(n)>int(v):
                    total_AA = 0
                    total_RA = 0
                    dup_vertice = deepcopy(vertice)
                    dup_vertice.remove(v)
                    for j in dup_vertice:
                        if adj[int(v)][int(j)] !=0 and adj[int(j)][int(n)] != 0:
                            Hn = Hindice[h][j]
                            if Hn==1:
                                pass
                            else:
                                total_AA = total_AA+1/math.log(2,Hn)
                                total_RA = total_RA+1/Hn
                    AA_info[(v,n)] = total_AA
                    RA_info[(v,n)] = total_RA
                    if (v,n) not in set(AA_info.keys()):
                        AA_info[(v,n)] = 0
                    if (v, n) not in set(AA_info.keys()):
                        AA_info[(v, n)] = 0
        Hn_AA_info[h] = AA_info
        Hn_RA_info[h] = RA_info
    print("AA_RA_Keys:",list(AA_info.keys()))
    return Hn_AA_info,Hn_RA_info


#传入所有节点集合vertice，得到每个节点跟需要预测的节点之间的共同邻居有哪些
def getCommonNeighbor_node(G,cur,vertice):
    commom = {}
    set_ver = set(vertice)
    for v in vertice:
        notDirecLinked = set_ver-set(cur[v])
        for n in notDirecLinked:
            if int(v)<int(n):
                commom[(v,n)] = sorted(nx.common_neighbors(G,v,n))
    return commom

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

def LRW():
    pass

G=nx.Graph()
v=getVertice(edges)
tr,te=devide_train_and_test_set(edges,0.5)
net = dictNet(tr,v)
print("net:",net)
print("te:",te)
G=nx.Graph(net)
nodes= set(G.nodes())
cur = dictNet(edges,v)
getNoLinked(cur,len(v))
degree=getDegree(G)
c=getCommonNeighbor_node(G,net,v)
print("common:",c)
getEdgeForPredict(net,v)
AA,RA=AA_RA(G)
print("AA:",AA)
print("RA:",RA)
core=getCore(G)
print("core_myself:",core,type(core))
print("core_package:",nx.core_number(G),type(nx.core_number(G)))
h=getHIndice(net,degree,core)
print("H",h)
CN_num(ad)
print()

# for i in H_AA:
#     print(i)
#     print(H_AA[i])
#     auc_AA=getAUC(te,H_AA[i])
#     print("AUC_AA_{index}:".format(index=i),auc_AA)
# for j in H_RA:
#     print(j)
#     print(H_RA[j])
#     auc_RA=getAUC(te,H_RA[j])
#     print("AUC_RA_{index}:".format(index=j),auc_RA)
plt.figure()
nx.draw(G,with_labels=True)
plt.show()