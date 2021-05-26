# from kay.mtsoo import *
from mtsoo import *
from slgep_lib.chromosome import *


def mfeaii_mgp(functions, config, callback=None, normal_beta=False):
    # unpacking hyper-parameters
    K = len(functions)                 # number of function
    N = config['pop_size'] * K         # population size
    # D = config['dimension']            # dimention size
    T = config['num_iter']             # number of iteration
    par = config['num_par']                # number of parents
    # sbxdi = config['sbxdi']            # number used in sbx-crossover

    pmdi = config['pmdi']
    pswap = config['pswap']
    rmp_matrix = np.zeros([K, K])

    # for sl_gep decode
    max_arity = config['max_arity']
    h_main = config['h_main']
    h_adf = config['h_adf']
    no_terminal = config['num_terminal']
    no_main = config['num_main']
    no_adf = config['num_adf']

    # initialize
    population = Slgep_pop(no_adf, no_terminal, no_main,
                           h_main, max_arity, h_adf)
    population.initialize(2*N)

    # evaluate
    for i in range(2 * N):
        sf = population[i].sf
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
            par = min(N-k, par)
            pl = population[k:k+par]
            sf = skill_factor[k:k+par]

            if normal_beta:
                # calculating beta equal posibilistic distribution.
                bl = np.ones((par, D))
                for i in range(0, par):
                    bl[i] *= learn_model(subpops[sf[i]],
                                         D).density(subpops[sf[i]])
            else:
                # alternative beta
                bl = np.random.normal(0.7, 0.1, size=(par, D))

            # Crossover
            # Check if chosen parent have same skill factor
            same_factor = True
            for i in range(par-1):
                if sf[i] != sf[i+1]:
                    same_factor = False

            if same_factor:
                cl = multiple_crossover(pl, bl)
                # mutate children
                for i in range(par):
                    cl[i] = mutate(cl[i], pmdi)

                cl = variable_swap_mul(cl, pswap)
                for i in range(par):
                    skill_factor[k:k+par] = sf

            # if chosen parent have different skill factor,
            else:
                # calculate max rmp between the first parent and the rest
                # if random < greatest rmp of the parents => perform cross-factorial crossover
                max_rmp = np.max(
                    np.array([rmp_matrix[sf[0], sf[i]] for i in range(1, par)]))

                if np.random.rand() < max_rmp:
                    cl = multiple_crossover(pl, bl)
                    for i in range(par):
                        cl[i] = mutate(cl[i], pmdi)
                    cl = variable_swap_mul(cl, pswap)

                for i in range(par):
                    sf_assign = np.random.randint(0, par)
                    skill_factor[k:k+par] = sf[sf_assign]

                # else perform crossover on random individual with the same skill factor as p1
                else:
                    for i in range(1, par):
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
                    for i in range(par):
                        cl[i] = mutate(cl[i], pmdi)
                    cl = variable_swap_mul(cl, pswap)
                    for i in range(par):
                        skill_factor[k:k+par] = sf

            # replace parents with children
            for i in range(par):
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
