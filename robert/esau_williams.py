import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from datetime import datetime
import sys
import os

if len(sys.argv) != 2:
    print('Please provide the cost of cable')
    exit(1)

start = datetime.now()
print("Start", start)

folder = f'./{start.strftime("%Y-%m-%d-%H-%M")}'
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

arr = np.loadtxt('./PS10.csv', delimiter=';')
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
    last_cost = calc_cost(graph)
    run = True
    run_counter = 0
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
            degree_of_i = graph.degree[vertex]
            degree_of_central_link = graph.degree[central_link_of_i]

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
                degree_of_j = graph.degree[vertex_j]

                trade_off_cable = - central_link_cost + edge_cost
                trade_off_connection = \
                    - get_connection_type(degree_of_i)[0] \
                    - get_connection_type(degree_of_j)[0] \
                    - get_connection_type(degree_of_central_link)[0] \
                    + get_connection_type(degree_of_i + 1)[0] \
                    + get_connection_type(degree_of_j + 1)[0] \
                    + get_connection_type(degree_of_central_link - 1)[0]

                trade_off_current = trade_off_cable + trade_off_connection

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

        graph.remove_edge(0, central_link_to_remove)
        graph.add_edge(start_vertex, end_vertex,
                       weight=adjacency_matrix[start_vertex][
                           end_vertex])

        new_cost = calc_cost(graph)

        if new_cost > last_cost:
            graph.remove_edge(start_vertex, end_vertex)
            graph.add_edge(0, central_link_to_remove,
                           weight=adjacency_matrix[0][central_link_to_remove])
            run = False
        else:
            last_cost = new_cost
            subtrees[subtree_index_of_end] += subtrees[subtree_index_of_start]
            subtrees.pop(subtree_index_of_start)

        if run_counter % 50 == 0:
            print('new cost:', new_cost)
            draw_graph(graph, run_counter)
            save_graph(graph, subtrees, run_counter)

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
