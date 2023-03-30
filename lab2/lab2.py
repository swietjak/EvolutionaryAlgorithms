import math
import random
import numpy as np
import time

n = 250
Tmax = 500
mi = 0.8
PARAMS_AMOUNT = 3
tau1 = 1 / (math.sqrt(2 * n))
tau2 = 1 / (math.sqrt(2 * math.sqrt(n)))


def load_file(number):
    with open(f"./data/model{number}.txt", "r") as f:
        inputs, outputs = [], []
        lines = f.readlines()
        for line in lines:
            i, o = line.strip().replace("  ", " ").split(" ")
            inputs.append(float(i))
            outputs.append(float(o))

        return inputs, outputs


inputs, expected_outputs = load_file(1)


def function(i, a, b, c):
    return a * ((i ** 2) - (b * math.cos(c * math.pi * i)))
    # return np.multiply(a, (np.subtract(np.power(i, 2), (np.multiply(b, math.cos(np.multiply(c, np.multiply(math.pi, i))))))))


def get_mse(computed_outputs):
    return sum([(expected_outputs[i] - computed_output) ** 2 for i, computed_output in enumerate(computed_outputs)]) / len(computed_outputs)


def get_outputs_from_population_member(population_member):
    return [function(input, population_member[0], population_member[1], population_member[2]) for input in inputs]


def get_initial_parameters():
    limit = 10
    a, b, c = np.random.uniform(-limit, limit, PARAMS_AMOUNT)
    sigmaA, sigmaB, sigmaC = [np.random.uniform(
        0, limit) for _ in range(PARAMS_AMOUNT)]

    parent = (a, b, c, sigmaA, sigmaB, sigmaC)

    mse = get_mse(get_outputs_from_population_member(parent))
    return (parent, mse)


def generate_initial_set():
    return [get_initial_parameters() for _ in range(n)]


def update_parameters(parameters, sigmas):
    return parameters + sigmas


def get_new_sigma(initial_sigma: float, r1: float) -> float:
    r2 = tau2 * np.random.normal(0, 1)
    return initial_sigma * np.exp(r1) * np.exp(r2)


def get_new_sigmas_arr(old_sigmas):
    r1 = tau1 * np.random.normal(0, 1)
    return [get_new_sigma(old_sigma, r1) for old_sigma in old_sigmas]


def get_fittest(parents):
    sorted_parents = sorted(parents, key=lambda err: err[1])
    return sorted_parents[:int(n * mi)]


def get_child(population_member):
    random_coefs = [np.random.normal(0, sigma)
                    for sigma in population_member[3:6]]
    new_parameters = list(
        np.sum([population_member[:3], random_coefs], axis=0))

    new_sigmas = get_new_sigmas_arr(population_member[3:6])
    child = list(new_parameters) + new_sigmas
    mse = get_mse(get_outputs_from_population_member(child))
    return (child, mse)


def draw_two_indicies(range_end: int):
    index = int(random.random() * range_end)
    index2 = index
    while index2 == index:
        index = int(random.random() * range_end)
    return index, index2


def get_populations_diff(parents, offspring):
    range = 50  # int((1 - mi)*n)
    return abs(sum([mse for _, mse in parents[:range]]) - sum([mse for _, mse in offspring[:range]]))


def reproduce_by_sum(parents_with_mses, crossover_fn):
    fittest_parents = get_fittest(parents_with_mses)
    children = []
    while len(children) < n:
        index, index2 = draw_two_indicies(len(fittest_parents))
        children.append(
            (get_child(crossover_fn(fittest_parents[index][0], fittest_parents[index2][0]))))
    new_generation = sorted(children + fittest_parents, key=lambda x: x[1])[:n]
    population_diff = get_populations_diff(parents_with_mses, new_generation)
    return new_generation, population_diff


def reproduce_by_offspring(parents_with_mses, crossover_fn):
    fittest_parents = get_fittest(parents_with_mses)
    offspring = []
    while len(offspring) < 5*n:
        index, index2 = draw_two_indicies(len(fittest_parents))
        offspring.append(
            (get_child(crossover_fn(fittest_parents[index][0], fittest_parents[index2][0]))))
    sorted_offspring = sorted(offspring, key=lambda x: x[1])[:n]
    population_diff = get_populations_diff(parents_with_mses, offspring)

    return sorted_offspring, population_diff


def intermediate_crossover(parent1, parent2):
    return list(np.sum([parent1, parent2], axis=0) / 2)


def discrete_crossover(parent1, parent2):
    return [parent1[i] if np.random.uniform(0, 1) > 0.5 else parent2[i] for i in range(len(parent1))]


def dummy_crossover(source, dummy):
    return source


def write_to_file(data):
    f = open('means.txt', "a")
    f.write(data)
    f.close()


def evolve(crossover_fn, selection_fn, iteration=0):
    population = generate_initial_set()
    population_diff = np.inf
    iterations = 0

    while population_diff > 10**-5:
        population, population_diff = selection_fn(population, crossover_fn)
        print(population[0], population_diff)
        iterations += 1

    values_string = f"iteration: {iteration} n: {n}; mi: {mi}; selection: {selection_fn.__name__}, crossover: {crossover_fn.__name__}"
    result_string = f"The ultimate chad is: {population[0][0]} {population[0][1]}"

    res = f"{values_string}\n{result_string}\n"
    write_to_file(res)
    return population[0][1]


selections = [reproduce_by_offspring, reproduce_by_sum]
crossovers = [intermediate_crossover, discrete_crossover, dummy_crossover]
individuals = [100, 250, 300, 500]
percentages = [0.5, 0.7, 0.8, 0.9]


avg_times = {}
for selection in selections:
    for crossover in crossovers:
        for population_number in individuals:
            n = population_number
            tau1 = 1 / (math.sqrt(2 * n))
            tau2 = 1 / (math.sqrt(2 * math.sqrt(n)))
            for percentage in percentages:
                mi = percentage
                results = []
                times = []
                for i in range(10):
                    start = time.time()
                    results.append(evolve(crossover, selection, i))
                    times.append(time.time() - start)
                values_string = f"n: {n}; mi: {mi}; offspring_n = {int(mi * n)} ;selection: {selection.__name__}, crossover: {crossover.__name__}"
                result_string = f" mean_mse: {np.mean(results):.3f}; std_mse: {np.std(results):.3f}; mean_time: {np.mean(times):.3f}, std_time: {np.std(times):.3f}\n"
                write_to_file(values_string + result_string)
