import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

max_heliostats = 128
price_cable = 10

price_conductor = 10
color_conductor = 'green'

price_switch8 = 100
color_switch8 = 'yellow'

price_switch16 = 800
color_switch16 = 'orange'

arr = np.loadtxt('../PS10.csv', delimiter=';')
arr = np.insert(arr, 0, [0, 0], axis=0)

folder = './2022-06-23-20-15-37'
adj = np.load(f'{folder}/files/adj_full.npy')
subtrees = np.load(f'{folder}/files/subtrees_full.npy', allow_pickle=True)
res_graph = nx.read_weighted_edgelist(
    f'{folder}/files/graph_edgelist_full.txt',
    nodetype=int)

wfolder = f'{folder}/more-img'
try:
    os.makedirs(wfolder)
except FileExistsError:
    pass

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


def calc_cost(graph):
    connection_cost = 0
    for i in range(1, graph.number_of_nodes()):
        connection_cost += get_connection_type(graph.degree[i])[0]

    cable_cost = graph.size('weight') * price_cable
    return connection_cost + cable_cost


def draw_graph(graph, counter, nodelist):
    color_map = []
    for vertex in nodelist:
        if vertex == 0:
            color_map.append('blue')
        else:
            color_map.append(get_connection_type(graph.degree[vertex])[1])

    nx.draw(graph, arr, nodelist=nodelist,
            node_color=color_map, node_size=50, ax=ax)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')

    ax.set_title(f'Connection of heliostats of subtree {counter} using EW')
    fig.savefig(f"{wfolder}/graph_subtree_{counter}.png", dpi=300)
    ax.clear()


for i, subtree in enumerate(subtrees):
    draw_graph(res_graph, i + 1, subtree)

draw_graph(res_graph, 'without nodes', [0])
