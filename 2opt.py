# coding: utf-8
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

# CONSTANTS
print("start")
data_file = './PS10.csv'

try:
    input_file = open(data_file, 'r')
except (OSError, IOError):
    print("read wrong")
    sys.exit()
    # read data file
lines = input_file.readlines()
# define number of items
nbItems = len(lines) - 1

cities = np.array([[float(0), float(0), 0]])
for i in range(1, nbItems):
    curr_line = lines[i].split(';')
    cities = np.append(cities, [[float(curr_line[0]), float(curr_line[1]), i]], axis=0)

print("\n---INSTANCE---")
print(np.shape(cities))

cities1 = np.array([[300, 0],
                    [450, 100],
                    [500, 200],
                    [550, 300],
                    [600, 400],
                    [500, 500],
                    [400, 500],
                    [300, 400],
                    [200, 500],
                    [100, 500],
                    [0, 400],
                    [50, 300],
                    [100, 200],
                    [150, 100],
                    [200, 50]])

# cities = cities1

print(np.shape(cities1))

# 2-opt algo #                 
COUNT_MAX = 500


def cut_cities(cities):
    allcity = np.delete(cities, 0, axis=0)
    cities1 = np.array([[float(0), float(0), 0]])
    cities2 = np.array([[float(0), float(0), 0]])
    cities3 = np.array([[float(0), float(0), 0]])
    cities4 = np.array([[float(0), float(0), 0]])
    cities5 = np.array([[float(0), float(0), 0]])
    for i in range(len(allcity)):
        if allcity[i][1] < (-1.6) * allcity[i][0]:
            cities1 = np.append(cities1, [allcity[i]], axis=0)
        if allcity[i][1] < 1.6 * allcity[i][0]:
            cities5 = np.append(cities5, [allcity[i]], axis=0)
        if (allcity[i][1] >= (-1.6) * allcity[i][0]) and (allcity[i][1] < (-5.6) * allcity[i][0]):
            cities2 = np.append(cities2, [allcity[i]], axis=0)
        if (allcity[i][1] >= 1.6 * allcity[i][0]) and (allcity[i][1] < 5.6 * allcity[i][0]):
            cities4 = np.append(cities4, [allcity[i]], axis=0)
        if (allcity[i][1] >= (-5.6) * allcity[i][0]) and (allcity[i][1] >= 5.6 * allcity[i][0]):
            cities3 = np.append(cities3, [allcity[i]], axis=0)

    # print(np.shape(cities3))
    cutedcities = [cities1, cities2, cities3, cities4, cities5]
    return cutedcities


# (nicht mehr verwendet) Da die Eingabedaten, die Sie selbst erstellen, der beste Weg sein können, 
# erhalten Sie zunächst eine zufällige Route (eine beliebige machbare Lösung, die Sie wählen)
def get_random_path(best_path):
    random.shuffle(best_path)
    path = np.append(best_path, best_path[0])
    return path


# Berechnen des Abstands zwischen zwei Punkten
def calculate_distance(from_index, to_index):
    return np.sqrt(np.sum(np.power(cities[from_index] - cities[to_index], 2)))


# Berechnen Sie die Entfernung des gesamten Weges
def calculate_path_distance(path):
    sum = 0.0
    for i in range(1, len(path)):
        sum += calculate_distance(path[i - 1], path[i])
    return sum


# 获取随机的起始点还有中间的反转后的路径
# Ermitteln des zufälligen Startpunkts und des umgekehrten Pfads dazwischen
def get_reverse_path(path):
    start = random.randint(1, len(path) - 1)
    while True:
        end = random.randint(1, len(path) - 1)
        if np.abs(start - end) > 1:
            break

    if start > end:
        path[end: start + 1] = path[end: start + 1][::-1]
        return path
    else:
        path[start: end + 1] = path[start: end + 1][::-1]
        return path


# 比较两个路径的长短
# Vergleich der Länge von zwei Pfaden
def compare_paths(path_one, path_two):
    return calculate_path_distance(path_one) > calculate_path_distance(path_two)


# 不断优化，得到一个最终的最短的路径
# Kontinuierliche Optimierung, um einen endgültigen kürzesten Weg zu finden
def update_path(path):
    count = 0
    while count < COUNT_MAX:
        # print(count)
        process_bar(count, COUNT_MAX)
        reverse_path = get_reverse_path(path.copy())
        if compare_paths(path, reverse_path):
            #print("found better")
            count = 0
            path = reverse_path
        else:
            count += 1
    return path


def opt_2():
    best_path = nn_tsp(cities)
    path1 = update_path(best_path)
    show(path1)
    # path = alter_tour(path1)
    # show(path)


def nn_tsp(cities):
    cities2 = cities
    start = 0
    tour = np.array([start])
    unvisited = np.delete(cities2, 0, axis=0)
    n = len(cities2)
    tour1=[0]
    while len(unvisited) != 0:
        process_bar(n - len(unvisited), n)
        # C = nearest_neighbor(cities2[tour[-1]], unvisited, cities2)
        index_in_all, index_in_rest = nearest_neighbor(cities2[tour[-1]], unvisited, cities2)
        #np.append(tour, index_in_all)
        tour1=tour1+[index_in_all]
        # unvisited = np.setdiff1d(unvisited, cities2[C])
        unvisited = np.delete(unvisited, index_in_rest, axis=0)
    # tour = np.append(tour, 0)
    return tour1


def nearest_neighbor(current_point, unvisited, cities2):
    "Find the city in cities that is nearest to city A."

    index_in_rest = -1
    dis = distance(unvisited[0], current_point)
    for i in range(len(unvisited)):
        new_dis = distance(unvisited[i], current_point)
        # print(new_dis)
        if new_dis <= dis:
            dis = new_dis
            index_in_rest = i
    # print(index_in_rest)
    min_city = unvisited[index_in_rest]
    l = int(min_city[2])
    #print(l)
    # print(min_city)

    # for l in range(len(cities2)):
    # print(cities2[l])
    # if (cities2[l][0] == min_city[0]) and (cities2[l][1] == min_city[1]):

    return l, index_in_rest


def distance(A, B):
    "The distance between two points."
    return np.sqrt(np.sum(np.power(A - B, 2)))


def show(path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(cities[:, 0], cities[:, 1], 'o', color='red')
    for i in range(len(path)):
        ax.plot(cities[path[i], 0], cities[path[i], 1], color='blue')
    plt.show()


def showcity(city):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(city[:, 0], city[:, 1], 'o', color='red')
    # ax.plot(cities[path, 0], cities[path, 1], color='red')
    plt.show()


def process_bar(num, total):
    rate = float(num) / total
    ratenum = int(100 * rate)
    r = '\r[{}{}]{}%'.format('*' * ratenum, ' ' * (100 - ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()


def reverse_segment_if_better(tour, i, j):
    "If reversing tour[i:j] would make the tour shorter, then do it."
    # Given tour [...A-B...C-D...], consider reversing B...C to get [...A-C...B-D...]
    A, B, C, D = cities[tour[i]], cities[tour[i + 1]], cities[tour[j - 1]], cities[tour[j % len(tour)]]

    # Are old edges (AB + CD) longer than new ones (AC + BD)? If so, reverse segment.
    # print(distance(A,B))
    if distance(A, B) + distance(C, D) > distance(A, C) + distance(B, D):
        print("yes")
        tour[i:j + 1] = tour[j:i - 1:-1]


def alter_tour(tour):
    "Try to alter tour for the better by reversing segments."
    # print(len(tour))
    original_length = calculate_path_distance(tour)
    list = (np.sort(all_segments(len(tour))))[::-1]
    for (start, end) in tqdm(list):
        reverse_segment_if_better(tour, start, end)
    # If we made an improvement, then try again; else stop and return tour.
    if calculate_path_distance(tour) < original_length:
        return alter_tour(tour)
    return tour


def all_segments(N):
    "Return (start, end) pairs of indexes that form segments of tour of length N."
    return [(start, start + length)
            for length in range(N - 2, 0, -1)
            for start in range(1, N - length)]

def path_inpart(cities):
    cities_inpart= cut_cities(cities)
    path=[]
    for i in range(len(cities_inpart)):
        print("\n",i)
        path=path+[update_path(nn_tsp(cut_cities(cities)[i]))]
    return path

# opt_2()
# show(alter_tour(nn_tsp(cities)))
# print(all_segments(5))
# showcity(cut_cities(cities))
# showcity(cities)
show(path_inpart(cities))
