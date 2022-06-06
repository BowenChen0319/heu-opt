# coding: utf-8
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse import csr_matrix
import networkx as nx

# CONSTANTS
OVERWRITE_EXISTING_IMAGES = True
OVERWRITE_EXISTING_FILES = False

# PRICES
PRICE_CABLE_PER_METER = 10
PRICE_CONDUCTOR = 100
PRICE_8SWITCH = 800
PRICE_16SWITCH = 1500

csv_filename = '../PS10.csv'
fig_folder = './figs'
file_folder = './files'
np_filename = f'{file_folder}/PS10.npy'

heliostats = None

# LOGGING
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True


def convert_text_file():
    """Converts given csv file to npy file for faster loading"""
    heliostats_txt = np.loadtxt(
        csv_filename,
        delimiter=';',
        # dtype={
        #     'names': ('x', 'y'),
        #     'formats': ('f', 'f'),
        # }
    )
    # Insert absolver
    heliostats_txt = np.insert(heliostats_txt, 0, (0, 0), axis=0)
    np.save(np_filename, heliostats_txt)


def load_np_file():
    return np.load(np_filename)


def _file_exists(filename):
    return Path(filename).is_file()


def file_exists(filename):
    return False if OVERWRITE_EXISTING_FILES else _file_exists(filename)


def image_exists(filename):
    return False if OVERWRITE_EXISTING_IMAGES else _file_exists(filename)


def load_heliostats():
    if not file_exists(np_filename):
        logging.debug('NPY file not found, converting text file')
        convert_text_file()

    global heliostats
    heliostats = load_np_file()
    logging.info(f'{len(heliostats)} heliostats loaded')


def metric_distance(point_a, point_b):
    x_a, y_a = point_a
    x_b, y_b = point_b
    return np.sqrt(np.power(x_b - x_a, 2) + np.power(y_b - y_a, 2))


def cost_distance(point_a, point_b):
    return metric_distance(point_a, point_b) * PRICE_CABLE_PER_METER


def create_image(graph, title, filename):
    filename = f'{fig_folder}/{filename}.png'
    if not image_exists(filename):
        fig, ax = plt.subplots(figsize=(8, 8))

        plt.title(title)
        nx.draw(graph, heliostats, ax, with_labels=False, node_size=25)

        ax.set_ylim(0, 800)
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.savefig(filename, dpi=300)
        plt.clf()


def create_init_graph():
    filename = f'{file_folder}/fully_connected_graph.txt'
    if file_exists(filename):
        logging.info('Reading graph from file. Please wait....')
        graph = nx.read_weighted_edgelist(filename)
    else:
        logging.info('Creating graph from matrix. Please wait....')
        adj_matrx = csr_matrix(
            squareform(pdist(heliostats) * PRICE_CABLE_PER_METER))

        graph = nx.from_scipy_sparse_array(adj_matrx, parallel_edges=False)
        nx.write_weighted_edgelist(graph, filename)
    logging.info('Graph loading completed.')
    return graph


def main():
    load_heliostats()
    if heliostats is None:
        logging.error('Failed to load heliostats')
        exit(1)

    # Create graph with only nodes
    graph = nx.Graph()
    graph.add_nodes_from(range(len(heliostats)))
    create_image(graph, 'Layout of heliostats including absorber', 'heliostats')

    # Let the real work begin
    graph = create_init_graph()


if __name__ == '__main__':
    main()
