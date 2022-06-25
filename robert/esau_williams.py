import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from datetime import datetime
import sys
import os
import random

if len(sys.argv) != 2:
    print('Please provide the cost of cable')
    exit(1)

start = datetime.now()
print("Start", start)

folder = f'./{start.strftime("%Y-%m-%d-%H-%M-%S")}'
os.makedirs(folder)
os.makedirs(f'{folder}/figs')
os.makedirs(f'{folder}/files')

matplotlib.use("Agg")

max_heliostats = 128
price_cable = int(sys.argv[1])

price_conductor = 100
color_conductor = 'green'

price_switch8 = 800
color_switch8 = 'yellow'

price_switch16 = 1500
color_switch16 = 'orange'

arr = np.loadtxt('../PS10.csv', delimiter=';')
arr = np.insert(arr, 0, [0, 0], axis=0)

with open(f'{folder}/info.txt', 'w') as f:
    f.write(f'Program started: {start}\n')
    f.write(f'---------------\n')
    f.write(f'Conductor -> Cost: {price_conductor}, Color: {color_conductor}\n')
    f.write(f'Switch 8 -> Cost: {price_switch8}, Color: {color_switch8}\n')
    f.write(f'Switch 16 -> Cost: {price_switch16}, Color: {color_switch16}\n')
    f.write(f'Cable -> Cost per meter: {price_cable}\n')
    f.write(f'---------------\n')
    f.write(f'Number of nodes: {len(arr)}\n')
    f.write('Using Esau-Williams heuristic to connect the nodes.\n')
    f.write(f'---------------\n')

adjacency_matrix = squareform(pdist(arr))
adjacency_matrix[adjacency_matrix == 0] = None

fig, ax = plt.subplots(figsize=(8, 8))


def get_connection_type(degree):
    if degree < 3:
        return [price_conductor, color_conductor]
    if degree < 9:
        return [price_switch8, color_switch8]
    elif degree < 17:
        return [price_switch16, color_switch16]
    else:
        return [99999999999999, 'red']


def get_subtree_index(array, element):
    for i in range(len(array)):
        if element in array[i]:
            return i
    return None


def calc_cost(graph):
    connection_cost = 0
    for i in range(1, graph.number_of_nodes()):
        connection_cost += get_connection_type(graph.degree[i])[0]

    cable_cost = graph.size('weight') * price_cable
    return connection_cost + cable_cost


def save_graph(graph, subtrees, counter):
    nx.write_weighted_edgelist(graph, f'{folder}/files/graph_edgelist'
                                      f'_{counter}.txt')
    np.save(f'{folder}/files/adj_{counter}', adjacency_matrix)
    np.save(f'{folder}/files/subtrees_{counter}',
            np.asanyarray(subtrees, dtype=object))


def draw_graph(graph, counter):
    color_map = ['blue']
    for i in range(1, graph.number_of_nodes()):
        color_map.append(get_connection_type(graph.degree[i])[1])

    nx.draw(graph, arr, node_size=50, node_color=color_map, ax=ax)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')

    ax.set_title('Connection of heliostats using EW')
    t = plt.figtext(0, .01, f"total cost = {calc_cost(graph)}")

    fig.savefig(f"{folder}/figs/graph_{counter}.png", dpi=300)
    ax.clear()
    plt.gcf().texts.remove(t)


def custom_min(trade_offs):
    best_of = 3
    buffer = []
    for _ in range(best_of):
        elem = min(trade_offs)
        buffer.append(elem)
        trade_offs.remove(elem)

    return buffer[random.choice(range(best_of))]


def esau_williams():
    # Add nodes to Graph
    graph = nx.Graph()
    graph.add_nodes_from(range(len(arr)))

    # Create point to point network
    for counter, item in enumerate(adjacency_matrix[0]):
        if counter == 0:
            continue
        graph.add_edge(0, counter, weight=item)

    print('Init weight:', calc_cost(graph))

    subtrees = [[x] for x in range(1, graph.number_of_nodes())]

    working_vertex = np.array([x for x in range(1, graph.number_of_nodes())])
    # np.random.shuffle(working_vertex)

    for vertex_i in range(graph.number_of_nodes(), 1, -1):
        working_vertex = np.delete(working_vertex, 0)
        trade_offs = []
        # check if directly connected to center
        if not graph.has_edge(0, vertex_i):
            continue

        vertex_i_subtree_index = get_subtree_index(subtrees, vertex_i)
        vertex_i_subtree = subtrees[vertex_i_subtree_index]
        edge_cost_0_i = adjacency_matrix[0][vertex_i]

        if len(vertex_i_subtree) >= max_heliostats:
            continue

        for vertex_j in range(1, graph.number_of_nodes()):

            if vertex_j in vertex_i_subtree:
                # already in same subtree
                continue

            vertex_j_subtree_index = get_subtree_index(subtrees, vertex_j)
            vertex_j_subtree = subtrees[vertex_j_subtree_index]

            if len(vertex_i_subtree) + len(vertex_j_subtree) > max_heliostats:
                # connection vertex j and vertex i would violate the constraint
                continue

            edge_cost_i_j = adjacency_matrix[vertex_i][vertex_j]
            degree_j = graph.degree[vertex_j]

            trade_off_cable = - edge_cost_0_i + edge_cost_i_j
            trade_off_loc = - get_connection_type(degree_j)[0] \
                            + get_connection_type(degree_j + 1)[0]
            trade_off_j = trade_off_cable + trade_off_loc

            trade_offs.append((trade_off_j, vertex_j, vertex_j_subtree_index))

        try:
            # best_trade_off, _, _ = min(trade_offs)
            trade_off, vertex_j, subtree_index_of_j = custom_min(trade_offs)
            # trade_off, vertex_j, subtree_index_of_j = min(trade_offs)
        except ValueError:
            continue

        if trade_off >= 0:
            continue

        graph.remove_edge(0, vertex_i)
        graph.add_edge(vertex_i, vertex_j, weight=adjacency_matrix[vertex_i][
            vertex_j])

        subtrees[subtree_index_of_j] += subtrees[vertex_i_subtree_index]
        subtrees.pop(vertex_i_subtree_index)

    draw_graph(graph, 'full')
    save_graph(graph, subtrees, 'full')
    print('End weight:', calc_cost(graph))


esau_williams()
end = datetime.now()

with open(f'{folder}/info.txt', 'a') as f:
    f.write(f'Program ended: {end}\n')
    f.write(f'Program runtime: {end - start}\n')

print("END:", end)
print("Runtime:", end - start)
