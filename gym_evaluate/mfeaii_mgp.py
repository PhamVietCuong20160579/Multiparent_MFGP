# from kay.mtsoo import *
from mtsoo import *
from slgep_lib.SL_gep import *

from scipy.stats import norm

# Problem, stuck in local optimum


def mfeaii_mgp(envs, config, callback=None, normal_beta=True, const_rmp=True):
    # unpacking hyper-parameters
    K = len(envs.envs)                 # number of function
    N = config['pop_size'] * K         # population size
    T = config['num_iter']             # number of iteration
    no_par = config['num_par']                # number of parents
    rmp = config['rmp']                # use constant rmp
    rmp_matrix = np.zeros([K, K])
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
        if const_rmp == False:
            rmp_matrix = population.learn_rmp(subpops)

        # select pair to crossover
        for k in range(0, N, no_par):
            no_p = min(N-k, no_par)

            # extract parents from population
            parents = [population.pop[k+i] for i in range(no_p)]

            # set beta
            if normal_beta:
                # calculating beta from normal distribution.
                bl = np.random.normal(0.7, 0.1, size=(no_p, D))
            else:
                # calculating beta from posibilistic distribution.
                bl = np.ones((no_p, D))
                for i in range(0, no_p):
                    sf = population.pop[k+i].sf
                    cgene = np.array([p.gene for p in subpops[sf]])
                    mean = np.mean(cgene, axis=0)
                    std = np.std(cgene, axis=0)
                    for j in range(D):
                        if std[j] == 0:
                            bl[i][j] *= 1
                        else:
                            bl[i][j] *= norm.pdf(
                                parents[i].gene[j], loc=mean[j], scale=std[j])

            # set rmp
            if const_rmp == False:
                # calculate max rmp between the first parents and the rest
                # if random < greatest rmp of the parents => perform cross-factorial crossover
                max_rmp = np.max(
                    np.array([rmp_matrix[parents[0].sf, parents[i].sf] for i in range(1, no_p)]))
            else:
                # use const rmp
                max_rmp = rmp

            # Crossover
            # Check if chosen parents have same skill factor
            same_sf = True
            for i in range(no_p-1):
                if parents[i].sf != parents[i+1].sf:
                    same_sf = False

            # if all chosen parents have same skill factor
            if same_sf:
                if no_p > 2:
                    cl = population.crossover_mul(parents, bl)
                    # mutate children
                    for i in range(no_p):
                        cl[i] = population.mutate(cl[i], mr)
                        cl[i].sf = parents[0].sf
                elif no_p == 2:
                    c1, c2 = population.onepoint_crossover(
                        parents[0], parents[1])
                    c1 = population.frequency_based_mutate(c1, mr)
                    c2 = population.frequency_based_mutate(c2, mr)
                    # c1, c2 = variable_swap(c1, c2, pswap)
                    c1.sf = parents[0].sf
                    c2.sf = parents[0].sf
                else:
                    continue

            # if chosen parents have different skill factor,
            elif np.random.rand() < max_rmp:
                if no_p > 2:
                    cl = population.crossover_mul(parents, bl)
                    for i in range(no_p):
                        cl[i] = population.mutate(cl[i], mr)

                        # assign random skill factor from parents to child
                        sf_assign = [p.sf for p in parents]
                        cl[i].sf = np.random.choice(sf_assign)
                else:
                    c1, c2 = population.onepoint_crossover(
                        parents[0], parents[1])
                    c1 = population.mutate(c1, mr)
                    c2 = population.mutate(c2, mr)
                    # c1, c2 = variable_swap(c1, c2, pswap)
                    if np.random.rand() < 0.5:
                        c1.sf = parents[0].sf
                    else:
                        c1.sf = parents[1].sf
                    if np.random.rand() < 0.5:
                        c2.sf = parents[0].sf
                    else:
                        c2.sf = parents[1].sf

            # else perform crossover on random individual with the same skill factor as p1
            else:
                for i in range(1, no_p):
                    parents[i] = population.find_relative(parents[0].sf)

                if no_p > 2:
                    cl = population.crossover_mul(parents, bl)
                    for i in range(no_p):
                        cl[i] = population.mutate(cl[i], mr)
                        cl[i].sf = parents[0].sf
                else:
                    c1, c2 = population.onepoint_crossover(
                        parents[0], parents[1])
                    c1 = population.frequency_based_mutate(c1, mr)
                    c2 = population.frequency_based_mutate(c2, mr)
                    # c1, c2 = variable_swap(c1, c2, pswap)
                    c1.sf = parents[0].sf
                    c2.sf = parents[0].sf

            # replace parents with children
            for i in range(no_p):
                population.pop[N + k + i] = cl[i]

        # re-evaluate
        population.evaluate(envs)

        # sort
        population.sort()

        # optimization info
        if const_rmp == False:
            message = {'algorithm': 'mfeaii_mp',
                       'rmp': round(rmp_matrix[0, 1], 1)}
        else:
            message = {'algorithm': 'mfeaii_mp', 'rmp': rmp}
        results = population.get_optimization_results(t, message)
        if callback:
            callback(results)

        desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join(
            '{:0.6f}'.format(res.fun) for res in results), message)
        iterator.set_description(desc)
