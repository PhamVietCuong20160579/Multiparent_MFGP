from scipy.optimize import OptimizeResult
from scipy.optimize import fminbound
from scipy.stats import norm, truncnorm
from copy import deepcopy
import numpy as np
from collections import namedtuple

from .function_set import *

ChromosomeRange = namedtuple('ChromosomeRange', ('R1', 'R2', 'R3', 'R4'))
# | --- Function Set --- | --- ADF Set --- | --- ADF Terminal Set --- | --- Terminals --- |
# | 'function_set'       | 'adf_set'       | 'adf_terminal_set'       | 'terminal_set'    |
# |                      |                 |        (Variables)       |     (Inputs)      |
# 0 ---------------------| R1 -------------| R2 ----------------------| R3 ---------------| R4


# A node in the BTS tree
class Node():
    def __init__(self, index, func):
        self.index = index
        self.name = func['name']
        self.arity = func['arity']
        self.func = func['func']
        self.value = 0
        self.children = []

    def set_value(self, value):
        self.value = value

    def _print_tree(self, root):
        x = ''
        if root is not None:
            if len(root.children) == 2:
                x += '('
                x += self._print_tree(root.children[0])
                x += self.name
                x += self._print_tree(root.children[1])
                x += ')'
            elif len(root.children) == 1:
                x += '('
                x += self.name
                x += self._print_tree(root.children[0])
                x += ')'
            else:
                x += self.name
        return x

    def print_tree(self):
        return self._print_tree(self)


# A single Chromosome (an individual of the enviroment)
class Chromosome():
    def __init__(self, no_adf, no_terminal, no_main, h_main, max_arity, h_adf, no_task):

        # skill-factor, fatorial-cost, scalar-fitness setup (mfeaII index)
        self.sf = np.random.randint(0, no_task)
        self.factorial_cost = np.full(no_task, np.inf)
        self.scalar_fitness = np.inf

        # parameter setup
        self.max_arity = max_arity
        self.no_terminal = no_terminal
        self.no_main = no_main
        self.h_main = h_main
        self.h_adf = h_adf
        self.no_adf = no_adf

        self.l_main = h_main*(max_arity-1) + 1
        self.l_adf = h_adf*(max_arity-1) + 1

        self.main_length = (self.h_main + self.l_main)
        self.adf_length = (self.h_adf + self.l_adf)

        self.D = self.no_main*(self.main_length) + \
            self.no_adf*(self.adf_length)

        # self.mr = (self.adf_length)/self.D

        # symbol list setup
        self.function_set = create_function_set()

        self.terminal_set = create_terminal_set(
            no_terminal=no_terminal)

        self.adf_set = create_adfs_set(
            no_adf=self.no_adf, max_arity=self.max_arity)

        self.adf_terminal_set = create_adfs_terminal_set(
            max_arity=self.max_arity)

        # symbol range setup
        R1 = len(self.function_set)
        R2 = R1 + len(self.adf_set)
        R3 = R2 + len(self.adf_terminal_set)
        R4 = R3 + len(self.terminal_set)

        self.chromosome_range = ChromosomeRange(R1, R2, R3, R4)

        # self generate gene string
        self.gene = self.generate_continuos()

    def _get_feasible_range(self, i):
        R1, R2, R3, R4 = self.chromosome_range
        # gene at i belong to one of the given mains
        if i < self.no_main * (self.h_main + self.l_main):
            # Head of main: adf_set and function_set
            if (i % (self.h_main + self.l_main) < self.h_main):
                return 0, R2
            # Tail of main: terminal_set
            else:
                return R3, R4
        # gene at i belong to one of adfs
        if (i - self.no_main * (self.h_main + self.l_main)) % (self.h_adf + self.l_adf) < self.h_adf:
            # Head of ADF: function_set
            return 0, R1
        else:
            # Tail of ADF: adf_terminal_set
            return R2, R3

    # rescale from result of crossover_mul to feasible_range
    def _gene_rescale(self, i, old_low, old_high):
        new_min, new_max = 0, 1
        L = old_low
        H = old_high
        new_range = new_max - new_min
        old_range = H - L

        vp = self.gene[i]
        if vp > old_high or vp < old_low:
            print('boundary wrong')
        re = (((vp - L) * new_range) / old_range) + new_min
        return re

    def generate_continuos(self):
        gene = np.random.rand(self.D)
        return gene

    def _translate_discrete(self, start, length):
        g = np.ones(length)
        for i in range(length):
            low, high = self._get_feasible_range(start + i)
            k = 1
            while (k/(high-low)) < self.gene[start + i]:
                k += 1
            g[i] *= (k+low-1)
        return g.astype('int32')

    def generate_tree(self, start, length):
        symbol_list = self.function_set + self.adf_set + \
            self.adf_terminal_set + self.terminal_set

        g = self._translate_discrete(start, length).tolist()

        # create Root node
        root = Node(index=g[0], func=symbol_list[g[0]])

        queue = [root]
        g.pop(0)
        while len(queue) and len(g):
            parent = queue.pop(0)
            for i in range(parent.arity):
                node = Node(index=g[0], func=symbol_list[g[0]])

                queue.append(node)
                g.pop(0)
                parent.children.append(node)

        return root

    def calculate_node(self, node: Node, terminal_list):
        R1, R2, R3, R4 = self.chromosome_range
        # if node is in terminal set (input set)
        if node.index >= R3:
            node.set_value(terminal_list[node.index - R3])
        # if node is in adf terminal set
        elif node.index >= R2:
            node.set_value(terminal_list[node.index - R2])
        # if node is ether function or adf symbol
        else:
            # need to calculate children of node as parameter to calculate node
            parameter = []
            for child in node.children:
                self.calculate_node(child, terminal_list)
                parameter.append(child.value)

            # if node is function symbol
            if node.index < R1:
                try:
                    value = node.func(*parameter)
                except Exception as e:
                    print(e)
                    print(node.index)
                    print(node.name)
                node.set_value(value)
            # if node is adf function
            else:
                adf = self.adfs[node.index-R1]
                self.calculate_node(adf, parameter)
                node.set_value(adf.value)

    def get_action(self, terminal_list):
        # everytime get_action is call, it re-prase a tree based on its gene
        # index of main
        main_index = [(i*self.main_length) for i in range(self.no_main)]

        # if envs do not provided enough parameter, fill extra with 0
        terminal_list = np.hstack(
            [terminal_list, np.zeros(len(self.terminal_set) - len(terminal_list))])
        # index of adf
        adf_index = [(self.no_main*self.main_length + i*self.adf_length)
                     for i in range(self.no_adf)]

        # self generate main tree
        self.mains = [self.generate_tree(
            main, self.main_length) for main in main_index]
        # self generate adf tree
        self.adfs = [self.generate_tree(
            adf, self.adf_length) for adf in adf_index]

        result = []
        for main in self.mains:
            self.calculate_node(main, terminal_list)
            # if main.value > 99:
            #     main.value = 100
            result.append(main.value)
        # result = softmax(result)

        return np.argmax(result)


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

    # only permute first half
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

    # this only happen at pswap chance
    def onepoint_crossover(self, p1, p2, pswap):
        D = p1.D
        x = np.random.randint(0, D)
        c1 = deepcopy(p1)
        c2 = deepcopy(p2)
        if np.random.rand() < pswap:
            if np.random.rand() < 0.5:
                c1.gene[0: x] = p2.gene[0: x]
                c2.gene[0: x] = p1.gene[0: x]
            else:
                c1.gene[x: D] = p2.gene[x: D]
                c2.gene[x: D] = p1.gene[x: D]

        return c1, c2

    def sbx_crossover(self, p1, p2, sbxdi):
        D = p1.D
        cf = np.empty([D])
        u = np.random.rand(D)

        cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (sbxdi + 1)))
        cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (sbxdi + 1)))

        c1 = deepcopy(p1)
        c2 = deepcopy(p2)

        c1.gene = 0.5 * ((1 + cf) * p1.gene + (1 - cf) * p2.gene)
        c2.gene = 0.5 * ((1 + cf) * p2.gene + (1 - cf) * p1.gene)

        c1.gene = np.clip(c1.gene, 0, 1)
        c2.gene = np.clip(c2.gene, 0, 1)
        return c1, c2

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
                # L, H = cl[j]._get_feasible_range(i)
                L, H = 0, 1
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

    # a pdf based mutation, inspired from the frequency based mutation
    def pdf_based_mutation(self, p, mutation_rate):
        sf = p.sf
        mr = self._calculate_mr(p, mutation_rate)
        sub = self.get_subpops()[sf]
        con_genes = np.array([p.gene for p in sub])
        mean = np.mean(con_genes, axis=0)
        std = np.std(con_genes, axis=0)
        c = deepcopy(p)

        for i in range(p.D):
            if np.random.rand() < mr[i]:
                # if the data is too focus, then take a random number from 0 to 1
                c.gene[i] = np.random.normal(loc=mean[i], scale=std[i])
                c.gene[i] = np.clip(c.gene[i], 0, 1)
        return c

    def mutate(self, p, mutation_rate, pmdi):
        c = deepcopy(p)
        mr = self._calculate_mr(p, mutation_rate)
        # for i in range(p.D):
        #     if np.random.rand() < mr[i]:
        #         c.gene[i] = np.random.rand()
        u = np.random.uniform(size=[p.D])

        for i in range(p.D):
            if np.random.rand() < mr[i]:
                if u[i] < 0.5:
                    delta = (2*u[i]) ** (1/(1+pmdi)) - 1
                    c.gene[i] = p.gene[i] + delta * p.gene[i]
                else:
                    delta = 1 - (2 * (1 - u[i])) ** (1/(1+pmdi))
                    c.gene[i] = p.gene[i] + delta * (1 - p.gene[i])
        c.gene = np.clip(c.gene, 0, 1)
        return c

    def find_relative(self, sf):
        subpop = [p for p in self.pop if p.sf == sf]
        return np.random.choice(subpop)

    def evaluate_individual(self, p, envs):
        agent = p
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
                    for j in range(N):
                        prob[j] *= norm.pdf(subpop[j].gene[d],
                                            loc=self.mean[d], scale=self.std[d])
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
