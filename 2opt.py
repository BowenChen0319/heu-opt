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
    # print(cities[from_index])
    # print(cities[to_index])
    return np.sqrt(np.sum(np.power(cities[from_index] - cities[to_index], 2)))


# Berechnen Sie die Entfernung des gesamten Weges
def calculate_path_distance(path):
    sum = 0.0
    for i in range(1, len(path)):
        sum += calculate_distance(path[i - 1], path[i])
    return sum

def c(tour, i, j):
    # print(tour[i])
    # print(tour[j])
    return calculate_distance(tour[i], tour[j])

def cal_diff(route,i,j):
    if j+1 == len(route):
        diff = c(route, i, i + 1) + c(route, j, 0) - c(route, i, j) - c(route, i + 1, 0)
    else:
        diff=c(route, i, i + 1) + c(route, j, j + 1) - c(route, i, j) - c(route, i + 1, j + 1)

    return diff


def swap(path, i, j):
    start = i
    end =j
    if start > end:
        #return path[:i] + list(reversed(path[i:j + 1])) + path[j + 1:]
        path[end: start + 1] = path[end: start + 1][::-1]
    else:
        #path = path[:i] + list(reversed(path[i:j + 1])) + path[j + 1:]
        path[start: end + 1] = path[start: end + 1][::-1]

    return path



def two_opt(route):
    best = route
    improved = True
    h = 0
    while improved:
        improved = False
        for i in tqdm(range(0, len(route) - 3)):

            for j in range(i + 1, len(route) - 1):
                if j - i == 1:
                    continue  # changes nothing, skip then
                # new_route = route[:]
                # print("\n", new_route)
                # new_route[i:j] = route[j-1:i-1:-1]  # this is the 2woptSwap
                # print("\n", new_route)
                diff=cal_diff(route,i,j)
                # print(gain)
                # if calculate_path_distance(new_route) < calculate_path_distance(best):  # what should cost be?
                if diff > 0:
                    # best = new_route

                    # best = swap(route, i + 1, j)
                    #best = get_reverse_path_with_index(route, i + 1, j)
                    best=swap(route,i+1,j)

                    route=best
                    improved = True
                    break  # return to while
                    #continue
            if improved:
                break
        route = best
        h = h + 1
        print("\n", h)

    return best



def nn_tsp(cities):
    cities2 = cities
    start = 0
    tour = np.array([start])
    unvisited = np.delete(cities2, 0, axis=0)
    n = len(cities2)
    tour1 = [0]
    while len(unvisited) != 0:
        process_bar(n - len(unvisited), n)
        # C = nearest_neighbor(cities2[tour[-1]], unvisited, cities2)
        index_in_all, index_in_rest = nearest_neighbor(cities2[tour[-1]], unvisited, cities2)
        # np.append(tour, index_in_all)
        tour1 = tour1 + [index_in_all]
        # unvisited = np.setdiff1d(unvisited, cities2[C])
        unvisited = np.delete(unvisited, index_in_rest, axis=0)
    # tour = np.append(tour, 0)
    return tour1


def path_to_tour(path):
    return path + [0]


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
    # print(l)
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
    if len(path) >= 2:
        for i in range(len(path)):
            ax.plot(cities[path[i], 0], cities[path[i], 1], color='blue')
    else:
        print("singel path")
        ax.plot(cities[path, 0], cities[path, 1], color='blue')
    plt.show()


def showsingle(path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(cities[:, 0], cities[:, 1], 'o', color='red')
    ax.plot(cities[path, 0], cities[path, 1], color='blue')
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


def path_inpart(cities):
    cities_inpart = cut_cities(cities)
    path = []
    for i in range(len(cities_inpart)):
        print("\n", i)
        path = path + [(nn_tsp(cut_cities(cities)[i]))]
    return path


# opt_2()
# show(alter_tour(nn_tsp(cities)))
# print(all_segments(5))
# showcity(cut_cities(cities))
# showcity(cities)
# show(path_inpart(cities))
showsingle(two_opt(nn_tsp(cut_cities(cities)[0]) + [0]))
#showsingle(two_opt(nn_tsp(cities) + [0]))
