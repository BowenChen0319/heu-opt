import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from datetime import datetime

matplotlib.use("Agg")

max_heliostats = 128
price_cable = 10
price_conductor = 10
price_switch8 = 100
price_switch16 = 800

arr = np.loadtxt('./PS10.csv', delimiter=';')
arr = np.insert(arr, 0, [0, 0], axis=0)

# arr = [[0,0], [1,1], [0,1], [1,0], [1,2], [2,1]]

adjacency_matrix = squareform(pdist(arr))
adjacency_matrix[adjacency_matrix == 0] = None


def draw_graph(graph, counter, weight):
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw(graph, arr, node_size=50, ax=ax)
    plt.figtext(0, .95, f"weight = {weight}")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')
    fig.savefig(f"./figs/graph_{counter}.png", dpi=300)


def get_subtree_index(array, element):
    for i in range(len(array)):
        if element in array[i]:
            return i
    return None


def esau_williams():
    # better implementation: https://github.com/vikramgopali1970/Esau-Williams-Algorithm-CMST/blob/master/EsauWilliamsAlgorithm.java

    # Add nodes to Graph
    graph = nx.Graph()
    graph.add_nodes_from(range(len(arr)))

    # Create point to point network
    for counter, item in enumerate(adjacency_matrix[0]):
        if counter == 0:
            continue
        graph.add_edge(0, counter, weight=item)

    print('Init weight:', graph.size('weight'))

    subtrees = [[x] for x in range(1, graph.number_of_nodes())]
    last_weight = graph.size('weight')
    run = True
    run_counter = 0
    print("Start", datetime.now())
    while run:
        run_counter += 1
        trade_offs = []

        for vertex in range(1, len(adjacency_matrix[0])):
            trade_offs_of_i = []

            subtree_index_of_i = get_subtree_index(subtrees, vertex)
            subtree_of_i = subtrees[subtree_index_of_i]
            if len(subtree_of_i) >= max_heliostats:
                continue

            central_link_of_i = subtree_of_i[0]

            for vertex_j, cost_i_j in enumerate(adjacency_matrix[vertex]):
                if vertex_j == 0:
                    continue

                if vertex_j in subtree_of_i:
                    continue

                subtree_index_of_j = get_subtree_index(subtrees, vertex_j)
                subtree_of_j = subtrees[subtree_index_of_j]

                if len(subtree_of_i) + len(subtree_of_j) > max_heliostats:
                    continue

                edge_cost = cost_i_j
                central_link_cost = adjacency_matrix[0][central_link_of_i]

                trade_off_current = - central_link_cost + edge_cost
                trade_offs_of_i.append((
                    trade_off_current,
                    vertex_j,
                    subtree_index_of_j,
                ))

            try:
                trade_off, vertex_j, subtree_index_of_j = min(
                    trade_offs_of_i)
            except ValueError:
                continue

            trade_offs.append((
                trade_off,
                vertex,
                vertex_j,
                subtree_index_of_i,
                subtree_index_of_j,
            ))

        try:
            best_trade_off, start_vertex, end_vertex, \
            subtree_index_of_start, subtree_index_of_end = min(trade_offs)
        except ValueError:
            break

        if start_vertex == end_vertex:
            continue

        if subtree_index_of_start is None or subtree_index_of_end is None:
            break

        central_link_to_remove = subtrees[subtree_index_of_start][0]

        subtrees[subtree_index_of_end] += subtrees[subtree_index_of_start]
        subtrees.pop(subtree_index_of_start)

        graph.remove_edge(0, central_link_to_remove)
        graph.add_edge(start_vertex, end_vertex,
                       weight=adjacency_matrix[start_vertex][
                           end_vertex])

        new_weight = graph.size('weight')

        if new_weight > last_weight:
            graph.remove_edge(start_vertex, end_vertex)
            graph.add_edge(0, central_link_to_remove,
                           weight=adjacency_matrix[0][central_link_to_remove])
            run = False
        else:
            last_weight = new_weight

        if run_counter % 50 == 0:
            print('new weight:', new_weight)
            draw_graph(graph, run_counter, new_weight)
        # draw_graph(graph)

    draw_graph(graph, 'full', graph.size('weight'))
    print('End weight:', graph.size('weight'))


esau_williams()
print("END: ", datetime.now())
