from process.PreReadFile import readFile as RF
import process.LinkPrediction as LP
import networkx as nx


if __name__ == "__main__":
    Yeast_path = r"C:\Users\Tang\Desktop\data\Yeast.txt"
    edges=RF(Yeast_path)
    tr, te = LP.devide_train_and_test_set(edges, 0.1)
    print("te:", te)
    # G.add_edges_from(tr)
    # nodes = set(G.nodes())
    v = set(LP.getVertice(edges))
    net = LP.dictNet(tr, v)
    G = nx.Graph(net)
    # adj = createAdj(v, net)
    # discard_node = v-nodes
    # print("discard_nodes:", discard_node)
    # G.add_nodes_from(discard_node)
    degree = LP.getDictDegree(G)
    core = LP.getCore(G)
    AA, RA = LP.AA_RA(G)
    print("AUC_AA:",LP.getAUC(te, AA))
    print("AUC_RA:",LP.getAUC(te, RA))
    h = LP.getHIndice(net, degree, core)
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
    eforPre = LP.getEdgeForPredict(net,v)
    P = LP.creatPMatrix(net,degree)
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
    Hn_Pmatrix = LP.getNPMatrixDict(P,5)
    NProbMatrix = LP.getNVerTransProbMatrixDict(Hn_Pmatrix, len(v))
    HSRW = LP.getHSRW(h['h(1)'], eforPre, NProbMatrix, 5)
    auc_HSRW = LP.getAUC(te, HSRW)
    print("auc_HSRW:", auc_HSRW)
    print("=================================Calculating Hn_HSRW======================================")
    Hn_HSRW = LP.getHn_HSRW(h, eforPre, NProbMatrix, len(v))
    for n in Hn_HSRW:
        print("auc_Hn_HSRW_{n}".format(n=n), ":", LP.getAUC(te, Hn_HSRW[n]))


