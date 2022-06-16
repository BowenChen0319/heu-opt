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

# arr = [[0,0], [1,1], [0,1], [1,0], [1,2], [2,1]]

adjacency_matrix = squareform(pdist(arr))

def draw_graph(graph):
    nx.draw(graph, arr, node_size=50, with_labels=True)
    plt.show()

def esau_williams(adjacency_matrix): # better implementation: https://github.com/vikramgopali1970/Esau-Williams-Algorithm-CMST/blob/master/EsauWilliamsAlgorithm.java
    # Add nodes to Graph
    graph = nx.Graph()
    graph.add_nodes_from(range(len(arr)))

    # Create star graph
    for counter, item in enumerate(adjacency_matrix[0]):
        if item == 0.0:
            continue
        
        graph.add_edge(0, counter, weight=item)
    
    draw_graph(graph)

    print('Init weight:', graph.size('weight'))

    for vertixI in range(1, graph.number_of_nodes()):
        print('Working on  vertices', vertixI)
        savings = []

        cost0I = nx.shortest_path_length(graph, 0, vertixI, weight='weight')
        
        for vertixJ in range(1, graph.number_of_nodes()):
            if vertixI == vertixJ:
                savings.append(999999)
                continue
            
            cost0J = nx.shortest_path_length(graph, 0, vertixJ, weight='weight')
            costIJ = adjacency_matrix[vertixI][vertixJ]
            savings.append(max(cost0I, cost0J) - costIJ)
        
        best_saving = min(savings)
        vertixJ = savings.index(best_saving)
        print(savings, best_saving, vertixJ)

        if vertixI == vertixJ:
            continue

        cost0J = nx.shortest_path_length(graph, 0, vertixJ, weight='weight')
        if cost0I < cost0J:
            edges = list(graph.edges(vertixJ, data=True))
        else:
            edges = list(graph.edges(vertixI, data=True))

        print(edges)
        cost = [options['weight'] for (_, _, options) in edges]
        startVertix, endVertix, _ = edges[cost.index(max(cost))]
        print('Removed Edge', startVertix, endVertix)
        graph.remove_edge(startVertix, endVertix)
        print('Added Edge', vertixI, vertixJ)
        graph.add_edge(vertixI, vertixJ, weight=adjacency_matrix[vertixI][vertixJ])

        draw_graph(graph)
    
    print('New weight:', graph.size('weight'))
    draw_graph(graph)
        


esau_williams(adjacency_matrix)