from process.PreReadFile import readFile as RF
import process.LinkPrediction as LP
import networkx as nx



if __name__ == "__main__":
    Jazz_path = r"C:\Users\Tang\Desktop\data\Jazz.txt"
    edges=RF(Jazz_path)
    print("========================================== Running Jazz network ==============================================")
    print("total:", len(edges))
    print(edges)
    v = LP.getVertice(edges)
    v = sorted([int(i) for i in v])
    print("Vertice:", len(v))
    print(v)
