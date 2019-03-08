from process.PreReadFile import readFile as RF
from process.PreReadFile import writeFile as WF
import process.LinkPrediction as LP
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx

net_path = [
    # # linux 下文件路径
    # r"/home/wsc331/LP/data/Yeast.txt",
    # r"/home/wsc331/LP/data/Food Webs/Florida.txt",
    # r"/home/wsc331/LP/data/Adolescent.txt",
    # r"/home/wsc331/LP/data/KingJames.txt",
    # r"/home/wsc331/LP/data/USAir.txt",
    # r"/home/wsc331/LP/data/Jazz.txt",
    # r"/home/wsc331/LP/data/Celegans.txt",
    # r"/home/wsc331/LP/data/NetScience.txt",
    # r"/home/wsc331/LP/data/Politicalblogs.txt",
    # r"/home/wsc331/LP/data/Power.txt",
    # r"/home/wsc331/LP/data/Router.txt",
    # r"/home/wsc331/LP/data/SciMet.txt",

    # windows 下文件路径
    # r"C:\Users\Tang\Desktop\data\Celegans.txt",
    # r"C:\Users\Tang\Desktop\data\KingJames.txt",
    # r"C:\Users\Tang\Desktop\data\NetScience.txt",
    # r"C:\Users\Tang\Desktop\data\Politicalblogs.txt",
    # r"C:\Users\Tang\Desktop\data\Power.txt",
    # r"C:\Users\Tang\Desktop\data\Router.txt",
    # r"C:\Users\Tang\Desktop\data\SciMet.txt",
    # r"C:\Users\Tang\Desktop\data\Food Webs\Florida.txt",
    # r"C:\Users\Tang\Desktop\data\Yeast.txt",
    # r"C:\Users\Tang\Desktop\data\USAir.txt",
    # r"C:\Users\Tang\Desktop\data\Jazz.txt",
    # r"C:\Users\Tang\Desktop\data\Adolescent.txt",
]

savefig_path=r"E:\MechinLearning\LinkPrediction\figure"
# linux 数据存储路径
savedata_path = r"/home/wsc331/LP/figure/NetscienceAndRouter.txt"
# #windows 数据存储路径
# savedata_path = r'E:\MechinLearning\LinkPrediction\figure\data.txt'

if __name__ == "__main__":
    for number in range(len(net_path)):
        path = net_path[number]
        edges=RF(path)
        name = path.split("\\")[-1].split(".")[0]
        print("============================== Running {name} network ===============================".format(name=name))
        WF(savedata_path,"============== {name} network ==============".format(name=name)+"\n")
        #定义最大步数step
        step = 16
        #定义画图需要的数据
        LRW_plot = []
        Hn_LRW_plot = {}
        SRW_plot = []
        HSRW_plot = []
        Hn_HSRW_plot = {}
    #    run_num = 5
    #    for num in range(run_num):
        tr, te = LP.devide_train_and_test_set(edges, 0.1)
        print("len_edges:",len(edges))
        print("len_trainSet:",len(tr))
        # G.add_edges_from(tr)
        # nodes = set(G.nodes())
        v = set(LP.getVertice(edges))
        print("len_v:",len(v))
        net = LP.dictNet(tr, v)
        G = nx.Graph(net)
        CN = LP.getCommonNeighbor_node(G, net, v)
        CN_info = {}
        for item in CN:
            CN_info[item] = len(CN[item])
        AUC_CN = LP.getAUC(te, CN_info)
        print("AUC_CN:",AUC_CN)
        WF(savedata_path,"AUC_CN:")
        WF(savedata_path,AUC_CN)
        # adj = createAdj(v, net)
        degree = LP.getDictDegree(G)
        core = LP.getCore(G)
        AA, RA = LP.AA_RA(G)
        AUC_AA = LP.getAUC(te, AA)
        print("AUC_AA:",AUC_AA)
        WF(savedata_path,"AUC_AA:")
        WF(savedata_path,AUC_AA)
        AUC_RA = LP.getAUC(te, RA)
        print("AUC_RA:",AUC_RA)
        WF(savedata_path,"AUC_RA:"+"\n")
        WF(savedata_path,AUC_RA)
        iter_num,h = LP.getHIndice(net, degree, core)
        # print("HnIndex", h)
        print("intra_num:",iter_num)
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
        for s in range(2,step+1):
            print("-------------------------step{n}-----------------------------".format(n=s))
            hn_LRW_plot = []

            NP = LP.getNPmatrix(P,s)
            probMatrix = LP.getVerTransProbMatrix(NP)
            lrw = LP.LRW(eforPre,degree,len(tr),probMatrix)
            auc = LP.getAUC(te,lrw)
            print("AUC_LRW:",auc)
            # 将n步转移的 AUC_LRW 存入 LRW_plot 列表(其中 step 就是转移步数)
            LRW_plot.append(auc)
            Hn_LRW = LP.getHn_LRW(eforPre,h,len(tr),probMatrix)
            for k in Hn_LRW:
                print(k)
                auc_LRW = LP.getAUC(te, Hn_LRW[k])
                print("AUC_LRW_{index}:".format(index=k), auc_LRW)
                hn_LRW_plot.append(auc_LRW)
            print(hn_LRW_plot)
            Hn_LRW_plot[str(s)] = hn_LRW_plot

            print("=================================Calculating HSRW=========================================")
            Hn_Pmatrix = LP.getNPMatrixDict(P,s)
            NProbMatrix = LP.getNVerTransProbMatrixDict(Hn_Pmatrix, len(v))
            HSRW = LP.getHSRW(h['h(1)'], eforPre, NProbMatrix, s)
            auc_HSRW = LP.getAUC(te, HSRW)
            print("auc_HSRW:", auc_HSRW)
            HSRW_plot.append(auc_HSRW)
            print("=================================Calculating Hn_HSRW======================================")
            Hn_HSRW = LP.getHn_HSRW(h, eforPre, NProbMatrix, len(v))

            AUC_Hn_HSRW = []
            for n in Hn_HSRW:
                auc_hn_HSRW = LP.getAUC(te, Hn_HSRW[n])
                print("auc_Hn_HSRW_{n}".format(n=n), ":", auc_hn_HSRW)
                AUC_Hn_HSRW.append(auc_hn_HSRW)
            print(AUC_Hn_HSRW)
            Hn_HSRW_plot[str(s)] = AUC_Hn_HSRW
        # 以上LRW_plot,Hn_LRW_plot,SRW_plot,HSRW_plot,Hn_HSRW_plot中字典形式{步数:[auc_h(1),auc_h(1),...,auc_h(n)]}
        # print("LRW_plot")
        WF(savedata_path,"AUC_LRW:"+"\n")
        WF(savedata_path,LRW_plot)
        # print(LRW_plot)
        # print("Hn_LRW_plot")
        WF(savedata_path, "Hn_AUC_LRW:" + "\n")
        WF(savedata_path, Hn_LRW_plot)
        # print(Hn_LRW_plot)
        # print("HSRW_plot")
        WF(savedata_path, "AUC_HSRW:" + "\n")
        WF(savedata_path, HSRW_plot)
        # print(HSRW_plot)
        # print("Hn_HSRW_plot")
        WF(savedata_path, "AUC_Hn_HSRW:" + "\n")
        WF(savedata_path, Hn_HSRW_plot )
        # print(Hn_HSRW_plot)

        y_Hn_LRW_plotDict = {}
        y_Hn_HSRW_plotDict = {}

        # y_n_LRW_plotDict = {}
        # y_n_HSRW_plotDict = {}

        # 因为暂时不画图而将下面的代码注释掉，要画图时可直接解除注释
        # axis_x = np.linspace(2,step,endpoint=True,num=step-2+1)
        # print("len_axis:",len(axis_x))

        # plt.xticks(np.linspace(0,step,endpoint=True,num=step-0+1),fontsize=6)
        # plt.yticks(np.linspace(0.7,1,50),fontsize=4)
        # y_LRW = LRW_plot
        # print("len_y_LRW:",len(y_LRW))
        # plt.figure(1)
        # plt.plot(axis_x,y_LRW,"x",label = "LRW",linewidth=0.5)
        #
        # y_HSRW = HSRW_plot
        # plt.plot(axis_x,y_HSRW,"--",label = "HSRW",linewidth=0.5)

        # 处理成{1步:各个指标的h(1),2步:各个指标的h(2)...}便于画相同Hn跟,auc跟步数关系的图(因为暂时不画图而将下面的代码注释掉，要画图时可直接解除注释)
        # for i in range(iter_num+1):
        #     y_Hn_LRW_plot = []
        #     y_Hn_HSRW_plot = []
        #     for s in range(2,step+1):
        #         y_Hn_LRW_plot.append(Hn_LRW_plot[str(s)][i])
        #         print("y_Hn_LRW_plot")
        #         print(y_Hn_LRW_plot)
        #         y_Hn_HSRW_plot.append(Hn_HSRW_plot[str(s)][i])
        #         print("y_Hn_HSRW_plot")
        #         print(y_Hn_HSRW_plot)
        #     y_Hn_LRW_plotDict[str(i)] = y_Hn_LRW_plot
        #     y_Hn_HSRW_plotDict[str(i)] = y_Hn_HSRW_plot
        # print("y_Hn_LRW_plotDict:")
        # print(y_Hn_LRW_plotDict)
        # print("y_Hn_HSRW_plotDict")
        # print(y_Hn_HSRW_plotDict)
        # 因为暂时不画图而将下面的代码注释掉，要画图时可直接解除注释
        # linestyle = ['-', '-.', '2', 'p', '*', '+','v']
        # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        # style = []
        # for i in range(len(color)):
        #     for j in range(len(linestyle)):
        #         style.append(color[i] + linestyle[j] )
        # for h_lrw in y_Hn_LRW_plotDict:
        #     plt.plot(axis_x,y_Hn_LRW_plotDict[h_lrw],'{style}'.format(style=style.pop()),label = "h({index})_LRW".format(index=h_lrw),linewidth=0.5)
        # for h_hsrw in y_Hn_HSRW_plotDict:
        #     plt.plot(axis_x, y_Hn_HSRW_plotDict[h_hsrw],'{style}'.format(style=style.pop()),label = "h({index})_HSRW".format(index=h_hsrw),linewidth=0.5)
        # plt.legend(loc = "upper right",fontsize="x-small",ncol = 2)
        #
        # plt.xlabel("steps")
        # plt.ylabel("AUC")
        # plt.title(name)
        # plt.savefig(savefig_path+"\\{name}_step.pdf".format(name=name))

        # #画Hn_indice的AUC跟迭代次数n的关系(因为暂时不画图而将下面的代码注释掉，要画图时可直接解除注释)
        # linestyle = ['-', '-.', '2', 'p', '*', '+', 'v']
        # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        # style = []
        # for i in range(len(color)):
        #     for j in range(len(linestyle)):
        #         style.append(color[i] + linestyle[j] )
        # axis_n = np.linspace(0,iter_num,endpoint=True,num=iter_num-0+1)
        # plt.figure()
        #
        # plt.xticks(np.arange(0,iter_num+1,1),fontsize=6)
        # plt.yticks(np.linspace(0.7, 1, 50),fontsize=6)
        #
        # for n_LRW in Hn_LRW_plot:
        #     plt.plot(axis_n,Hn_LRW_plot[n_LRW],'{style}'.format(style=style.pop()),label="{s}step_LRW".format(s=n_LRW),linewidth=0.5)
        # for n_HSRW in Hn_HSRW_plot:
        #     plt.plot(axis_n, Hn_LRW_plot[n_HSRW],'{style}'.format(style=style.pop()),label="{s}step_HSRW".format(s=n_HSRW), linewidth=0.5)
        # plt.legend(loc="upper right", fontsize="x-small", ncol=2)
        #
        # plt.xlabel("n")
        # plt.ylabel("AUC")
        # plt.title(name)
        # plt.savefig(savefig_path + "\\{name}_n.pdf".format(name=name))


