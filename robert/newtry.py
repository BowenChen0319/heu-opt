import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import networkx as nx

max_heliostats = 128
price_cable = 10
price_conductor = 10
price_switch8 = 100
price_switch16 = 800

arr = np.loadtxt('./PS10.csv', delimiter=';')
arr = np.insert(arr, 0, [0,0], axis=0)

arr = [[0,0], [1,1], [0,1], [1,0], [1,2], [2,1]]

adjacency_matrix = squareform(pdist(arr))

def draw_graph(graph):
    nx.draw(graph, arr, node_size=50, with_labels=True)
    plt.show()

def get_min_element(arr):
    return min(arr, key=lambda x: x['cost'])

graph = nx.Graph()
graph.add_nodes_from(range(len(arr)))

def a(graph):
    tradeoff = []
    for vertixI in range(1, graph.number_of_nodes()):
        tradeoffI = []
        for vertixJ in range(1, graph.number_of_nodes()):
            if vertixI == vertixJ:
                continue

            tradeoffI.append({'vertixJ': vertixJ, 'cost': adjacency_matrix[vertixI][vertixJ]})
        
        elem = get_min_element(tradeoffI)
        tradeoff.append({'vertixI': vertixI, 'vertixJ': elem['vertixJ'], 'cost': elem['cost']})
    
    elem = get_min_element(tradeoff)
    graph.add_edge(elem['vertixI'], elem['vertixJ'], weight=elem['cost'])
    draw_graph(graph)

a(graph)
a(graph)
a(graph)
a(graph)

# https://d-nb.info/1124508023/34 read....
# https://sites.pitt.edu/~dtipper/2110/CMST_example.pdf