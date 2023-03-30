import numpy as np
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt

P = 100
n = 0.5
pm = 0.1
Tmax = 1000


def parse_axis(axis: str, letter):
    arr = axis.replace(f'{letter} = [', '').replace(';', '').replace(']', '')

    return [float(num) for num in arr.split(' ')]


def get_points(fileName: str) -> np.ndarray:
    with open(f'./cities_data/{fileName}.txt', 'r') as f:
        _, x, y = f.readlines()
        parsed_x = parse_axis(x, 'x')
        parsed_y = parse_axis(y, 'y')

        return np.array(list(zip(parsed_x, parsed_y)))


def get_distances_arr(points: np.ndarray) -> np.ndarray:
    return np.array([np.array([np.linalg.norm(p1 - p2) for p2 in points]) for p1 in points])


points = get_points("cities_4")
distances_array = get_distances_arr(points)
number_of_cities = len(points)


def is_in_array(perm: np.ndarray, perm_list: np.ndarray) -> bool:
    for created in perm_list:
        if np.array_equal(perm, created):
            return True
    return False


def get_distances_sum(permutation: np.ndarray) -> float:
    res = 0
    for i in range(len(permutation) - 1):
        res += distances_array[permutation[i], permutation[i+1]]
    res += distances_array[permutation[0], permutation[-1]]
    return res


def get_parents(previous_generation: dict, previous_generation_distances: np.ndarray) -> np.ndarray:
    max_distance = np.amax(previous_generation_distances)
    probabilities = np.array([(max_distance - dist) /
                              max_distance for dist in previous_generation_distances])
    probabilities_sum = np.sum(probabilities)
    probabilities = [
        1/P] * P if probabilities_sum == 0 else probabilities / probabilities_sum

    indicies = np.random.choice(range(P), int(P*n), p=probabilities)

    res = np.array([previous_generation[i] for i in indicies])

    return res


def get_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    o1 = [None] * len(parent1)
    index = 0
    while True:
        if o1[index] != None:
            break
        o1[index] = parent1[index]

        index = np.where(parent2 == parent1[index])[0][0]
    for index in range(len(parent2)):
        if o1[index] == None:
            o1[index] = parent2[index]
    return o1


def draw_two_indicies(range_end: int):
    index = int(random.random() * range_end)
    index2 = index
    while index2 == index:
        index = int(random.random() * range_end)
    return index, index2


def mutate(offspring: np.ndarray) -> np.ndarray:
    probability = np.random.uniform(0, 1)
    if probability <= pm:
        index, index2 = draw_two_indicies(len(offspring))
        offspring[index], offspring[index2] = offspring[index2], offspring[index]


def get_child(parents):
    index, index2 = draw_two_indicies(int(P*n))
    off = get_crossover(
        parents[index], parents[index2])

    mutate(off)
    return off


def populate(parents: np.ndarray) -> np.ndarray:
    return np.array([get_child(parents) for _ in range(int(P * n))])


def save_result(res):
    f = open("results.txt", 'a')
    f.write(res)
    f.close()


def save_means(res):
    f = open("means.txt", 'a')
    f.write(res)
    f.close()


def evolve(iteration=0):
    population = np.array([np.random.permutation(10) for _ in range(P)])
    for i in range(Tmax):
        old_generation_distances = {index: get_distances_sum(
            p) for index, p in enumerate(population)}

        new_parents = get_parents(
            population, list(old_generation_distances.values()))
        new_generation = populate(new_parents)  # nowe permutacje punktów

        population = np.concatenate((population, new_generation), axis=0)
        population_distances = [(p, get_distances_sum(p)) for p in population]

        sorted_population = sorted(population_distances, key=lambda x: x[1])
        population = np.array([p[0] for p in sorted_population[:P]])

    values_string = f"iteration: {iteration} P: {P}; n: {n}; pm: {pm}"
    result_string = f"The ultimate score is: {population[0]} {get_distances_sum(population[0])}"

    res = f"{values_string}\n{result_string}\n"
    save_result(res)
    print(res)
    # draw_route(population[0])

    return get_distances_sum(population[0])


def draw_route(indicies):
    x = [points[i][0] for i in indicies]
    y = [points[i][1] for i in indicies]
    for index, (i, j) in enumerate(zip(x, y)):
        plt.text(i, j+0.5, f'({i}, {j})')
    for i in range(0, len(x)):
        plt.plot(x[i:i+2], y[i:i+2], 'bo-')
    plt.savefig("Route.png")
    plt.clear()


evolve()
P_values = [100, 250, 300, 500]
n_values = [0.5, 0.7, 0.8, 0.9]
pm_values = [0.1, 0.2, 0.3, 0.5]

for pvalue in P_values:
    P = pvalue
    for n_value in n_values:
        n = n_value
        for pm_value in pm_values:
            pm = pm_value
            sum = 0
            for iteration in range(10):
                sum += evolve(iteration)
            save_means(f"P: {P}; n: {n}; pm: {pm} mean: {sum / 10}\n")
