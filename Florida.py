from process.PreReadFile import readFile as RF
import process.LinkPrediction as LP
import networkx as nx

if __name__ == "__main__":
    Florida_path = r"C:\Users\Tang\Desktop\data\Food Webs\Florida.txt"
    e = RF(Florida_path)
    print(e)
    v=LP.getVertice(e)
    v = sorted([int(i) for i in v])
    print("Vertice:", len(v))
    print(v)