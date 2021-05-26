# from kay.mtsoo import *
from mtsoo import *
from slgep_lib.chromosome import *


def mfeaii_mgp(functions, config, callback=None, normal_beta=False):
    # unpacking hyper-parameters
    K = len(functions)                 # number of function
    N = config['pop_size'] * K         # population size
    D = config['dimension']            # dimention size
    T = config['num_iter']             # number of iteration
    par = config['num_par']                # number of parents
    # sbxdi = config['sbxdi']            # number used in sbx-crossover

    pmdi = config['pmdi']
    pswap = config['pswap']
    rmp_matrix = np.zeros([K, K])

    # initialize
    population = np.random.rand(2 * N, D)
    skill_factor = np.array([i % K for i in range(2 * N)])
    factorial_cost = np.full([2 * N, K], np.inf)
    scalar_fitness = np.empty([2 * N])

    # evaluate
    for i in range(2 * N):
        sf = skill_factor[i]
        factorial_cost[i, sf] = functions[sf](population[i])
    scalar_fitness = calculate_scalar_fitness(factorial_cost)

    # sort
    sort_index = np.argsort(scalar_fitness)[::-1]
    population = population[sort_index]
    skill_factor = skill_factor[sort_index]
    factorial_cost = factorial_cost[sort_index]

    # evolve
    iterator = trange(T)
    for t in iterator:
        # permute current population
        permutation_index = np.random.permutation(N)
        population[:N] = population[:N][permutation_index]
        skill_factor[:N] = skill_factor[:N][permutation_index]
        factorial_cost[:N] = factorial_cost[:N][permutation_index]
        factorial_cost[N:] = np.inf

        # learn rmp
        subpops = get_subpops(population, skill_factor, N)
        rmp_matrix = learn_rmp(subpops, D)

        # select pair to crossover
        for k in range(0, N, par):
            p = min(N-k, par)
            if p <= 1:
                continue
            pl = population[k:k+p]
            sf = skill_factor[k:k+p]

            if normal_beta:
                # calculating beta equal posibilistic distribution.
                bl = np.ones((p, D))
                for i in range(0, p):
                    bl[i] *= learn_model(subpops[sf[i]],
                                         D).density(subpops[sf[i]])
            else:
                # alternative beta
                bl = np.random.normal(0.7, 0.1, size=(p, D))

            # Crossover
            # Check if chosen parent have same skill factor
            same_factor = True
            for i in range(p-1):
                if sf[i] != sf[i+1]:
                    same_factor = False

            # If chosen parents have same skill factor
            if same_factor:
                cl = multiple_crossover(pl, bl)
                # mutate children
                for i in range(p):
                    cl[i] = mutate(cl[i], pmdi)

                cl = variable_swap_mul(cl, pswap)
                for i in range(p):
                    skill_factor[k+i] = sf[i]

            # if chosen parents have different skill factor,
            else:
                # calculate max rmp between the first parent and the rest
                # if random < greatest rmp of the parents => perform cross-factorial crossover
                max_rmp = np.max(
                    np.array([rmp_matrix[sf[0], sf[i]] for i in range(1, p)]))

                if np.random.rand() < max_rmp:
                    cl = multiple_crossover(pl, bl)
                    for i in range(p):
                        cl[i] = mutate(cl[i], pmdi)
                    cl = variable_swap_mul(cl, pswap)

                for i in range(p):
                    sf_assign = np.random.randint(0, p)
                    skill_factor[k+i] = sf[sf_assign]

                # else perform crossover on random individual with the same skill factor as p1
                else:
                    for i in range(1, p):
                        pl[i] = find_relative(
                            population, skill_factor, sf[0], N)
                        while True:
                            for j in range(0, i):
                                if (pl[i] == pl[j]).all():
                                    pl[i] = find_relative(
                                        population, skill_factor, sf[0], N)
                                    continue
                            break

                    cl = multiple_crossover(pl, bl)
                    for i in range(p):
                        cl[i] = mutate(cl[i], pmdi)
                    cl = variable_swap_mul(cl, pswap)
                    for i in range(p):
                        skill_factor[k+i] = sf[i]

            # replace parents with children
            for i in range(p):
                population[N + k + i, :] = cl[i]

        # evaluate
        for i in range(N, 2 * N):
            sf = skill_factor[i]
            factorial_cost[i, sf] = functions[sf](population[i])
        scalar_fitness = calculate_scalar_fitness(factorial_cost)

        # sort
        sort_index = np.argsort(scalar_fitness)[::-1]
        population = population[sort_index]
        skill_factor = skill_factor[sort_index]
        factorial_cost = factorial_cost[sort_index]

        best_fitness = np.min(factorial_cost, axis=0)
        c1 = population[np.where(skill_factor == 0)][0]
        c2 = population[np.where(skill_factor == 1)][0]
        scalar_fitness = scalar_fitness[sort_index]

        # optimization info
        message = {'algorithm': 'mfeaii_mp', 'rmp': round(rmp_matrix[0, 1], 1)}
        results = get_optimization_results(
            t, population, factorial_cost, scalar_fitness, skill_factor, message)
        if callback:
            callback(results)

        desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join(
            '{:0.6f}'.format(res.fun) for res in results), message)
        iterator.set_description(desc)
