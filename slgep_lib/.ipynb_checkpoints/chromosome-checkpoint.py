import numpy as np
from copy import deepcopy
from collections import namedtuple

from numpy.core.fromnumeric import size
from . import function_set

ChromosomeRange = namedtuple('ChromosomeRange', ('R1', 'R2', 'R3', 'R4'))
# | --- Function Set --- | --- ADF Set --- | --- ADF Terminal Set --- | --- Terminals --- |
# | 'function_set'       | 'adf_set'       | 'adf_terminal_set'       | 'terminal_set'    |
# |                      |                 |        (Variables)       |     (Inputs)      |
# 0 ---------------------| R1 -------------| R2 ----------------------| R3 ---------------| R4


class Node():
    def __init__(self, index, arity, parent):
        self.index = index
        self.arity = arity
        self.parent = parent
        self.children = []


class Chromosome():
    def __init__(self, gene):
        self.gene = gene
        self.sf = np.random.randint(0, 2)
        self.fatorial_cost = np.inf
        self.scalar_fitness = np.inf


class Population():
    def __init__(self, no_adf, no_terminal, no_main, h_main, max_arity, h_adf):

        # parameter setup
        self.max_arity = max_arity
        self.no_terminal = no_terminal
        self.no_main = no_main
        self.h_main = h_main
        self.h_adf = h_adf
        self.no_adf = no_adf

        self.l_main = h_main*(max_arity-1) + 1
        self. l_adf = h_adf*(max_arity-1) + 1
        self. D = no_main*(h_main+self.l_main) + no_adf*(h_adf+self.l_adf)
        self.mr = (h_adf + self. l_adf)/self. D

        # symbol list setup
        self.function_set = function_set.create_function_set()

        self.terminal_set = function_set.create_terminal_set(
            no_terminal=no_terminal)

        self.adf_set = function_set.create_adfs_set(
            no_adf=self.no_adf, max_arity=self.max_arity)

        self.adf_terminal_set = function_set.create_adfs_terminal_set(
            max_arity=self.max_arity)

        # symbol range setup
        R1 = len(self.function_set)
        R2 = R1 + len(self.adf_set)
        R3 = R2 + len(self.adf_terminal_set)
        R4 = R3 + len(self.terminal_set)
        self.pop = []

        self.chromosome_range = ChromosomeRange(R1, R2, R3, R4)

    def _get_feasible_range(self, i):
        R1, R2, R3, R4 = self.chromosome_range
        # gene at i belong to one of the given mains
        if i < self.no_main * (self.h_main + self.l_main):
            # Head of main: adf_set and function_set
            if (i % (self.h_main + self.l_main) <= self.h_main):
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

    def create_gene(self, D):
        gene = np.zeros(D)
        for index in range(D):
            low, high = self._get_feasible_range(index)
            gene[index] = np.random.rand(low, high)
        return gene

    def initialize(self, pop):
        self.pop = []
        pass

    def cross_over(self, p1, p2):
        pass

    def multation(self, p1):
        pass

    def cal_action(self, p1):
        pass
