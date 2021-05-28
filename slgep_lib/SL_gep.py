import numpy as np
from copy import deepcopy
from scipy.stats import norm
from scipy.special import softmax

from scipy.optimize import fminbound
from scipy.optimize import OptimizeResult

from .chromosome import Chromosome

# The Slgep population, consist of multiple Chromosome


class Slgep_pop():
    def __init__(self, no_adf, no_terminal, no_main, h_main, max_arity, h_adf, no_pop, no_task):

        # parameter setup
        self.max_arity = max_arity
        self.no_terminal = no_terminal
        self.no_main = no_main
        self.h_main = h_main
        self.h_adf = h_adf
        self.no_adf = no_adf
        self.no_task = no_task
        self.rmp_matrix = np.zeros([no_task, no_task])

        self.pop = self.initialize(no_pop, no_task)

    def initialize(self, no_pop, no_task):
        pop = []
        for _ in range(no_pop):
            gene = Chromosome(self.no_adf, self.no_terminal,
                              self.no_main, self.h_main, self.max_arity, self.h_adf, no_task)
            pop.append(gene)
        return pop

    def evaluate(self, envs):
        no_pop = len(self.pop)
        fc = []
        for i in range(no_pop):
            agent = self.pop[i]
            sf = agent.sf
            result = envs.run_env(sf, agent)
            agent.factorial_cost[sf] = result
            fc.append(agent.factorial_cost)
        # re-calculate s_fitness, based on current sf, not best sf, if best sf, need to re-assign sf
        # s_fitness = 1 / np.min(np.argsort(np.argsort(fc, axis=0), axis=0) + 1, axis=1)
        ranking = np.argsort(np.argsort(fc, axis=0), axis=0) + 1
        b_sf = np.argmin(ranking, axis=1)
        s_fitness = 1 / np.min(ranking, axis=1)

        # re-assign sf to ones agent perform best
        for i in range(no_pop):
            self.pop[i].scalar_fitness = s_fitness[i]
            self.pop[i].sf = b_sf[i]

    def sort(self):
        self.pop.sort(key=lambda x: x.scalar_fitness, reverse=True)

    # this is wrong, only permute first half
    def permute(self):
        N = int(len(self.pop)/2)
        self.pop[:N] = np.random.permutation(self.pop[:N]).tolist()

    def get_subpops(self):
        subpops = []
        for i in range(self.no_task):
            subpop = [p for p in self.pop if p.sf == i]
            subpops.append(subpop)
        return subpops

    def _learn_models(self, subpops):
        K = len(subpops)
        D = subpops[0][0].D
        models = []
        for k in range(K):
            subpop = subpops[k]
            models.append(self._learn_model(subpop, D))
        return models

    def _learn_model(self, subpop, D):
        num_sample = len(subpop)
        num_random_sample = int(np.floor(0.1 * num_sample))
        # rand_pop = np.random.rand(num_random_sample, D)
        rand_pop = self.initialize(num_random_sample, self.no_task)
        con_pops = rand_pop + subpop
        con_genes = np.array([p.gene for p in con_pops])
        mean = np.mean(con_genes, axis=0)
        std = np.std(con_genes, axis=0)
        return Model(mean, std, num_sample, con_genes)

    def learn_rmp(self, subpops):
        K = self.no_task
        rmp_matrix = np.eye(K)
        models = self._learn_models(subpops)

        for k in range(K - 1):
            for j in range(k + 1, K):
                probmatrix = [np.ones([models[k].num_sample, 2]),
                              np.ones([models[j].num_sample, 2])]
                probmatrix[0][:, 0] = models[k].density(subpops[k])
                probmatrix[0][:, 1] = models[j].density(subpops[k])
                probmatrix[1][:, 0] = models[k].density(subpops[j])
                probmatrix[1][:, 1] = models[j].density(subpops[j])

                rmp = fminbound(lambda rmp: log_likelihood(
                    rmp, probmatrix, K), 0, 1)
                rmp += np.random.randn() * 0.01
                rmp = np.clip(rmp, 0, 1)
                rmp_matrix[k, j] = rmp
                rmp_matrix[j, k] = rmp

        return rmp_matrix

    def onepoint_crossover(self, p1, p2):
        D = p1.D
        x = np.random.randint(0, D)
        c1 = deepcopy(p1)
        c2 = deepcopy(p2)
        if np.random.rand() < 0.5:
            c1.gene[0: x] = p2.gene[0: x]
            c2.gene[0: x] = p1.gene[0: x]
        else:
            c1.gene[x: D] = p2.gene[x: D]
            c2.gene[x: D] = p1.gene[x: D]
        return c1, c2

    # will get out of convertable range, need to check
    def crossover_mul(self, pl, bl):
        no_par = len(pl)
        cl = deepcopy(pl)
        for j in range(no_par):
            for i in range(pl[j].D):
                cl[j].gene[i] = pl[j].gene[i] + bl[j][i] * \
                    (pl[(j+1) % no_par].gene[i] - pl[(j+2) % no_par].gene[i])

                L, H = cl[j]._get_feasible_range(i)

                old_low = (L + bl[j][i]*(L - H))

                old_high = (H + bl[j][i]*(H - L))

                cl[j].gene[i] = cl[j]._gene_rescale(i, old_low, old_high)

        return cl

    def crossover_mul_second(self, pl, bl, rmp_matrix):
        no_par = len(pl)
        cl = deepcopy(pl)
        rmp = np.ones((no_par, no_par))
        for i in range(no_par-1):
            for j in range(i, no_par):
                if pl[i].sf == pl[j].sf:
                    rmp[i][j] *= 1
                else:
                    rmp[i][j] *= rmp_matrix[pl[i].sf, pl[j].sf]

        for j in range(no_par):
            for i in range(pl[j].D):
                L, H = cl[j]._get_feasible_range(i)
                cl[j].gene[i] = (pl[j].gene[i] + bl[j, i] *
                                 (rmp[j][(j+1) % no_par] * pl[(j+1) % no_par].gene[i] -
                                  rmp[j][(j+2) % no_par] * pl[(j+2) % no_par].gene[i]))

                old_low = (L + bl[j][i]*(rmp[j][(j+1) % no_par]
                           * L - rmp[j][(j+2) % no_par] * H))

                old_high = (H + bl[j][i]*(rmp[j][(j+1) %
                            no_par] * H - rmp[j][(j+2) % no_par] * L))

                cl[j].gene[i] = cl[j]._gene_rescale(i, old_low, old_high)
        return cl

    def _calculate_mr(self, p, scaling_factor):
        x = p
        best, _ = self._get_best_individual_of_task(p.sf)
        a, b = np.random.choice(self.pop, 2)

        vp = np.ones(x.D)
        vp[np.where(x.gene == best.gene)] = 0

        th = np.ones(x.D)
        th[np.where(a.gene == b.gene)] = 0

        re = 1 - (1 - scaling_factor*vp)*(1 - scaling_factor*th)
        return re

    # a frequency based mutation, based on MFGP paper
    def frequency_based_mutate(self, p, mutation_rate):
        x = self.pop[0]
        genes = x.gene.reshape(1, x.D)
        for i in range(1, len(self.pop)):
            y = self.pop[i].gene.reshape((1, x.D))
            genes = np.concatenate((genes, y))

        k = np.random.randint(0, p.D)
        mr = self._calculate_mr(p, mutation_rate)
        c = deepcopy(p)
        for i in range(p.D):
            if (np.random.rand() < mr[i]) and (k != i):
                # get frequency of each value of gene
                values, counts = np.unique(genes[:, i], return_counts=True)
                s = np.argsort(counts)
                counts = counts[s]
                values = values[s]

                # frequency based random
                vp = np.random.randint(0, np.sum(counts))
                for j in range(len(counts)):
                    if vp > counts[j]:
                        vp -= counts[j]
                    else:
                        c.gene[i] = values[j]
                        break
        return c

    def mutate(self, p, mutation_rate):
        c = deepcopy(p)
        mr = self._calculate_mr(p, mutation_rate)
        for i in range(p.D):
            if np.random.rand() < mr[i]:
                low, high = p._get_feasible_range(i)
                c.gene[i] = np.random.randint(low, high)
        return c

    def find_relative(self, sf):
        subpop = [p for p in self.pop if p.sf == sf]
        return np.random.choice(subpop)

    # def variable_swap_mul(self, pl, pswap):
    #     parent = [self.pop[i] for i in pl]

    def evaluate_individual(self, p, envs):
        agent = self.pop[p]
        sf = agent.sf
        result = envs.run_env(sf, agent)
        return result

    def get_optimization_results(self, t, message):
        K = self.no_task
        N = len(self.pop) // 2
        results = []
        for k in range(K):
            result = OptimizeResult()
            x, fun = self._get_best_individual_of_task(k)
            result.x = x.gene
            result.fun = fun
            result.message = message
            result.nit = t
            result.nfev = (t + 1) * N
            results.append(result)
        return results

    def _get_best_individual_of_task(self, t):
        # select individuals from task sf
        subpop = [p for p in self.pop if p.sf == t]

        if subpop:
            # select best individual
            x = max(enumerate(subpop), key=lambda x: x[1].scalar_fitness)[1]
            fun = -x.factorial_cost[t]
            return x, fun
        # if there is no task that has sf of t, assign randomly some task to t
        else:
            # print('no individual have this sf {}'.format(t))
            switch_sf = np.random.choice(self.pop, int(len(self.pop)/2))
            for p in switch_sf:
                p.sf = t

            return self._get_best_individual_of_task(t)


# copy and modified from mstoo lib

class Model:
    def __init__(self, mean, std, num_sample, sample):
        self.mean = mean
        self.std = std
        self.num_sample = num_sample
        self.sample = sample

    def density(self, subpop):
        D = subpop[0].D
        N = len(subpop)
        prob = np.ones([N])
        subgene = np.array([p.gene for p in subpop])
        for d in range(D):
            # actually, the prob can get really low beaucause the number are integer randomed in very small area. (like [10-16])
            # and so does std.
            # therefore calculating can get so minor that its cause multiply underflow or exp underflow when take np.exp(x**2/2)

            # this is kind of questionable
            # actually safer to just use constant rmp
            if self.std[d] == 0:
                prob *= np.ones([N])
            else:
                try:
                    # prob *= norm.pdf(subgene[:, d],
                    #                  loc=self.mean[d], scale=self.std[d])
                    for j in range(N):
                        prob[j] *= norm.pdf(subpop[j].gene[d],
                                            loc=self.mean[d], scale=self.std[d])
                        if prob[j] < 1e-50:
                            prob[j] = 1e-50
                except Exception as e:
                    print('calculate density error')
                    print(e)
                    print(subgene[:, d])
                    print(self.mean[d])
                    print(self.std[d])
                    print(prob)
                    print(self.sample[:, d])
                    exit()
        return prob


def log_likelihood(rmp, prob_matrix, K):
    posterior_matrix = deepcopy(prob_matrix)
    value = 0
    for k in range(2):
        for j in range(2):
            if k == j:
                posterior_matrix[k][:, j] = posterior_matrix[k][:,
                                                                j] * (1 - 0.5 * (K - 1) * rmp / float(K))
            else:
                posterior_matrix[k][:, j] = posterior_matrix[k][:,
                                                                j] * 0.5 * (K - 1) * rmp / float(K)
        value = value + np.sum(-np.log(np.sum(posterior_matrix[k], axis=1)))
    return value
