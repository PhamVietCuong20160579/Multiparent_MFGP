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
        if root is not None:
            if len(root.children) == 2:
                self._print_tree(root.children[0])
                print(root.name)
                self._print_tree(root.children[1])
            elif len(root.children) == 1:
                print(root.name)
                self._print_tree(root.children[0])
            else:
                print(root.name)

    def print_tree(self):
        self._print_tree(self)


# A single Chromosome (an individual of the enviroment)
class Chromosome_discrete():
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
        self.gene = self.generate_gene()

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
        new_min, new_max = self._get_feasible_range(i)
        L = np.floor(old_low)
        H = np.ceil(old_high)
        new_range = new_max - new_min
        old_range = H - L

        vp = self.gene[i]
        re = (((vp - L) * new_range) / old_range) + new_min
        return re

    def generate_gene(self):
        gene = np.zeros(self.D)
        for index in range(self.D):
            low, high = self._get_feasible_range(index)
            gene[index] = np.random.randint(low, high)

        return gene.astype(np.int32)

    def generate_tree(self, start, stop):
        symbol_list = self.function_set + self.adf_set + \
            self.adf_terminal_set + self.terminal_set

        g = self.gene[start:stop].copy().tolist()

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
        try:
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
                    value = node.func(*parameter)
                    node.set_value(value)
                # if node is adf function
                else:
                    adf = self.adfs[node.index-R1]
                    self.calculate_node(adf, parameter)
                    node.set_value(adf.value)
        except Exception as e:
            print('calculate node error')
            print(e)
            print(terminal_list)
            print(node.index)
            print(self.chromosome_range)
            exit()

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
            main, main + self.main_length) for main in main_index]
        # self generate adf tree
        self.adfs = [self.generate_tree(
            adf, adf + self.adf_length) for adf in adf_index]

        result = []
        for main in self.mains:
            self.calculate_node(main, terminal_list)
            # if main.value > 99:
            #     main.value = 100
            result.append(main.value)
        # result = softmax(result)

        return np.argmax(result)


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
        new_min, new_max = self._get_feasible_range(i)
        L = np.floor(old_low)
        H = np.ceil(old_high)
        new_range = new_max - new_min
        old_range = H - L

        vp = self.gene[i]
        re = (((vp - L) * new_range) / old_range) + new_min
        return re

    def generate_continuos(self):
        gene = np.random.rand(self.D)
        return gene

    def _translate_discrete(self, start, length):
        g = np.ones(length)
        for i in range(length):
            low, high = self._get_feasible_range(start + i)
            k = 0
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
        try:
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
                    value = node.func(*parameter)
                    node.set_value(value)
                # if node is adf function
                else:
                    adf = self.adfs[node.index-R1]
                    self.calculate_node(adf, parameter)
                    node.set_value(adf.value)
        except Exception as e:
            print('calculate node error')
            print(e)
            print(terminal_list)
            print(node.index)
            print(self.chromosome_range)
            exit()

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
