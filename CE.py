from process.PreReadFile import readFile as RF
import process.LinkPrediction as LP
import networkx as nx


if __name__ == "__main__":
    Celegans_path = r"C:\Users\Tang\Desktop\data\Celegans.txt"
    edges = RF(Celegans_path)
    print("total:", len(edges))
    print(edges)
    v = LP.getVertice(edges)
    v = sorted([int(i) for i in v])
    print("startVertice:",v[0])
    print("Vertice:", len(v))
    print(v)


