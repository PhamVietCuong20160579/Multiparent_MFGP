# from kay.mtsoo import *
from mtsoo import *
from slgep_lib.chromosome_continuous import *

from scipy.stats import norm
# Problem, stuck in local optimum


def mfeaii_mgp(envs, config, callback=None):
    # unpacking hyper-parameters
    K = len(envs.envs)                 # number of function
    N = config['pop_size'] * K         # population size
    T = config['num_iter']             # number of iteration
    no_par = config['num_par']                # number of parents
    rmp_matrix = np.zeros([K, K])
    sbxdi = config['sbxdi']
    pmdi = config['pmdi']
    pswap = config['pswap']
    rmp_matrix = np.zeros([K, K])

    # for sl_gep decode
    max_arity = config['max_arity']
    h_main = config['h_main']
    h_adf = config['h_adf']
    # no_main = config['no_main']
    # no_adf = config['no_adf']
    # no_terminal = config['no_terminal']
    no_main = np.max([envs.envs[i].action_space.n for i in range(K)])
    no_adf = no_main*2
    no_terminal = np.max([envs.envs[i].reset().shape[0] for i in range(K)])

    # initialize
    population = Slgep_pop(no_adf, no_terminal, no_main,
                           h_main, max_arity, h_adf, no_pop=2*N, no_task=K)

    result_list = []

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
        # this step is skiped if we want to rank the population and then crossover
        population.permute()

        # learn rmp
        subpops = population.get_subpops()
        rmp_matrix = population.learn_rmp(subpops)

        # select pair to crossover
        for k in range(0, N, no_par):
            no_p = min(N-k, no_par)

            # extract parents from population
            parents = [population.pop[k+i] for i in range(no_p)]

            # calculating beta from posibilistic distribution.
            bl = np.ones((no_p, D))
            for i in range(0, no_p):
                sf = parents[i].sf
                concatenate_gene = np.array([p.gene for p in subpops[sf]])
                mean = np.mean(concatenate_gene, axis=0)
                std = np.std(concatenate_gene, axis=0)
                for j in range(D):
                    if std[j] == 0:
                        bl[i][j] *= 1
                    else:
                        bl[i][j] *= norm.pdf(
                            parents[i].gene[j], loc=mean[j], scale=std[j])

            # set rmp
            # calculate max rmp between the first parents and the rest
            # if random < greatest rmp of the parents => perform cross-factorial crossover
            max_rmp = np.max(
                np.array([rmp_matrix[parents[0].sf, parents[i].sf] for i in range(1, no_p)]))

            # Crossover
            # Check if chosen parents have same skill factor
            same_sf = True
            for i in range(no_p-1):
                if parents[i].sf != parents[i+1].sf:
                    same_sf = False

            # what happen if no_p is equal to 2 or 1????

            # if all chosen parents have same skill factor
            if same_sf:
                cl = deepcopy(parents)
                # crossover
                # cl = population.crossover_multiparent(parents, bl)
                for i in range(no_p-1):
                    cl[i], cl[(i+1)] = population.sbx_crossover(
                        parents[i], parents[(i+1)], sbxdi)

                # mutate children
                for i in range(no_p):
                    cl[i] = population.mutate(cl[i], pmdi)

                for i in range(no_p-1):
                    cl[i], cl[(i+1)] = population.variable_swap(
                        cl[i], cl[(i+1)], pswap)

                for i in range(no_p):
                    cl[i].sf = parents[0].sf

            # if chosen parents have different skill factor,
            elif np.random.rand() < max_rmp:
                cl = population.innertask_crossover_multiparent(
                    parents, bl, rmp_matrix)
                # ol = deepcopy(cl)
                for i in range(no_p):
                    cl[i] = population.mutate(cl[i], pmdi)
                    # ol[i], ol[(i+1) % no_p] = population.variable_swap(
                    #     cl[i], cl[(i+1) % no_p], pswap)

                    # assign random skill factor from parents to child
                    sf_assign = [p.sf for p in parents]
                    cl[i].sf = np.random.choice(sf_assign)

            # else perform crossover on random individual with the same skill factor as p1
            else:
                for i in range(1, no_p):
                    parents[i] = population.find_relative(parents[0].sf)

                cl = deepcopy(parents)
                # crossover

                # cl = population.crossover_multiparent(parents, bl)
                for i in range(no_p-1):
                    cl[i], cl[(i+1)] = population.sbx_crossover(
                        parents[i], parents[(i+1)], sbxdi)

                # mutate children
                for i in range(no_p):
                    cl[i] = population.mutate(cl[i], pmdi)

                for i in range(no_p-1):
                    cl[i], cl[(i+1)] = population.variable_swap(
                        cl[i], cl[(i+1)], pswap)

                for i in range(no_p):
                    cl[i].sf = parents[0].sf

            # replace parents with children
            for i in range(no_p):
                population.pop[N + k + i] = cl[i]

        # re-evaluate
        population.evaluate(envs)

        # sort
        population.sort()

        # optimization info
        message = {'algorithm': 'mfeaii_mp', 'rmp': round(rmp_matrix[0, 1], 1)}
        results = population.get_optimization_results(t, message)
        if callback:
            callback(results)

        desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join(
            '{:0.6f}'.format(res.fun) for res in results), message)
        iterator.set_description(desc)
        result_list.append(results)

    return result_list
