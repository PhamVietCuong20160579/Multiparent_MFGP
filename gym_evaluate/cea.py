from mtsoo import *
from slgep_lib.chromosome_continuous import *


def cea(envs, config, callback=None):
    # unpacking hyper-parameters
    K = len(envs.envs)                 # number of function
    N = config['pop_size'] * K         # population size
    T = config['num_iter']             # number of iteration
    mr = config['mutation_rate']

    # for sl_gep decode
    max_arity = config['max_arity']
    h_main = config['h_main']
    h_adf = config['h_adf']
    no_main = envs.envs[0].action_space.n
    no_adf = no_main*2
    # no_terminal = envs.envs[0].reset().shape[0]
    no_terminal = np.max([envs.envs[i].reset().shape[0] for i in range(K)])
    # no_terminal = config['num_terminal']
    # no_main = config['num_main']
    # no_adf = config['num_adf']

    # initialize
    population = Slgep_pop(no_adf, no_terminal, no_main,
                           h_main, max_arity, h_adf, no_pop=2*N, no_task=K)
    # dimention size
    D = population.pop[0].D

    # evaluate
    population.evaluate(envs)

    # sort
    population.sort()

    # evolve
    iterator = trange(T)
    for t in iterator:
        # permute current population
        population.permute()

        # select pair to crossover
        for i in range(0, N, 2):

            # extract parent
            p1 = population.pop[i]
            sf1 = p1.sf
            p2 = population.find_relative(sf1)
            # recombine parent
            c1, c2 = population.onepoint_crossover(p1, p2)
            c1 = population.mutate(c1, mr)
            c2 = population.mutate(c2, mr)
            # save child
            c1.sf = sf1
            c2.sf = sf1
            population.pop[N+i], population.pop[N+i+1] = c1, c2

        # evaluate
        population.evaluate(envs)

        # sort
        population.sort()

        # c1 = population[np.where(skill_factor == 0)][0]
        # c2 = population[np.where(skill_factor == 1)][0]

        # optimization info
        message = {'algorithm': 'cea'}
        results = population.get_optimization_results(t, message)
        if callback:
            callback(results)

        desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join(
            '{:0.6f}'.format(res.fun) for res in results), message)
        iterator.set_description(desc)

    return results
