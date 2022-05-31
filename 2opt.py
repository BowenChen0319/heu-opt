# coding: utf-8
import sys

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

cities = np.array([[float(0), float(0)]])
for i in range(1, nbItems):
    curr_line = lines[i].split(';')
    cities = np.append(cities, [[float(curr_line[0]), float(curr_line[1])]], axis=0)


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
        #print(count)
        process_bar(count,COUNT_MAX)
        reverse_path = get_reverse_path(path.copy())
        if compare_paths(path, reverse_path):
            count = 0
            path = reverse_path
        else:
            count += 1
    return path


def opt_2():
    best_path = nn_tsp(cities)
    path = update_path(best_path)
    show(path)


def nn_tsp(cities):
    
    cities2 = cities
    start = 0
    tour = np.array([start])
    unvisited = np.delete(cities2, 0, axis=0)
    n = len(cities2)

    while len(unvisited) != 0:
        process_bar(n-len(unvisited),n)
        #C = nearest_neighbor(cities2[tour[-1]], unvisited, cities2)
        index_in_all, index_in_rest = nearest_neighbor(cities2[tour[-1]], unvisited, cities2)
        tour = np.append(tour, index_in_all)
        #unvisited = np.setdiff1d(unvisited, cities2[C])
        unvisited = np.delete(unvisited, index_in_rest, axis=0)
    return tour


def nearest_neighbor(A, unvisited, cities2):
    "Find the city in cities that is nearest to city A."

    
    min_city_index = -1
    dis = distance(unvisited[0], A)
    for i in range(len(unvisited)):
        new_dis = distance(unvisited[i], A)
        # print(new_dis)
        if new_dis <= dis:
            dis = new_dis
            min_city_index = i
    #print(min_city_index)
    min_city = unvisited[min_city_index]
    #print(min_city)

    for l in range(len(cities2)):
        #print(cities2[l])
        if (cities2[l][0] == min_city[0]) and (cities2[l][1] == min_city[1]):
            return l, min_city_index


def distance(A, B):
    "The distance between two points."
    return np.sqrt(np.sum(np.power(A - B, 2)))


def show(path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(cities[:, 0], cities[:, 1], 'o', color='red')
    ax.plot(cities[path, 0], cities[path, 1], color='red')
    plt.show()

def process_bar(num, total):
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r[{}{}]{}%'.format('*'*ratenum,' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()


opt_2()
#show(nn_tsp(cities))
