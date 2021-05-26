from mtsoo import *
from slgep_lib.chromosome import *


def mfeaii(envs, config, callback=None):
    # unpacking hyper-parameters
    K = len(envs.envs)                 # number of function
    N = config['pop_size'] * K         # population size
    T = config['num_iter']             # number of iteration
    # rmp = config['rmp']                # use constant rmp
    mr = config['mutation_rate']

    # for sl_gep decode
    max_arity = config['max_arity']
    h_main = config['h_main']
    h_adf = config['h_adf']
    no_main = envs.envs[0].action_space.n
    no_adf = no_main*2
    no_terminal = np.max([envs.envs[i].reset().shape[0] for i in range(K)])
    # no_terminal = config['num_terminal']
    # no_main = config['num_main']
    # no_adf = config['num_adf']

    # initialize
    population = Slgep_pop(no_adf, no_terminal, no_main,
                           h_main, max_arity, h_adf, no_pop=2*N, no_task=K)
    # evaluate
    population.evaluate(envs)

    # evaluate
    population.evaluate(envs)

    # sort
    population.sort()

    # evolve
    iterator = trange(T)
    for t in iterator:
        # permute current population
        population.permute()

        # learn rmp
        subpops = population.get_subpops()
        rmp_matrix = population.learn_rmp(subpops)

        # select pair to crossover
        for i in range(0, N, 2):
            p1, p2 = population.pop[i], population.pop[i + 1]
            sf1, sf2 = p1.sf, p2.sf

            # crossover
            if sf1 == sf2:
                c1, c2 = population.onepoint_crossover(p1, p2)
                c1 = population.frequency_based_mutate(c1, mr)
                c2 = population.frequency_based_mutate(c2, mr)
                # c1, c2 = variable_swap(c1, c2, pswap)
                c1.sf = sf1
                c2.sf = sf1
            elif sf1 != sf2 and np.random.rand() < rmp_matrix[sf1, sf2]:
                c1, c2 = population.onepoint_crossover(p1, p2)
                c1 = population.mutate(c1, mr)
                c2 = population.mutate(c2, mr)
                # c1, c2 = variable_swap(c1, c2, pswap)
                if np.random.rand() < 0.5:
                    c1.sf = sf1
                else:
                    c1.sf = sf2
                if np.random.rand() < 0.5:
                    c2.sf = sf1
                else:
                    c2.sf = sf2
            else:
                p2 = population.find_relative(sf1)

                c1, c2 = population.onepoint_crossover(p1, p2)
                c1 = population.frequency_based_mutate(c1, mr)
                c2 = population.frequency_based_mutate(c2, mr)
                # c1, c2 = variable_swap(c1, c2, pswap)
                c1.sf = sf1
                c2.sf = sf1

            # replace parent with child
            population.pop[N+i], population.pop[N+i+1] = c1, c2

        # evaluate
        population.evaluate(envs)

        # sort
        population.sort()

        # optimization info
        message = {'algorithm': 'mfeaii', 'rmp': round(rmp_matrix[0, 1], 1)}
        results = population.get_optimization_results(t, message)
        if callback:
            callback(results)

        desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join(
            '{:0.6f}'.format(res.fun) for res in results), message)
        iterator.set_description(desc)
