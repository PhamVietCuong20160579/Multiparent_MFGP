from copy import deepcopy
import pickle
import yaml
import os


class Saver:

    def __init__(self, config, instance, seed):
        '''Folder result/instance
                            config.yaml
                            <seed>.pkl
        Parameters
        ----------
            config (dict): configuration of the problem
            instance (str): name of the benchmark
        '''
        self.seed = seed
        self.instance = instance
        # Create result folder
        folder = 'result'
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = 'result/%s' % instance
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Save configuration
        path = os.path.join(folder, 'config.yaml')
        _config = deepcopy(config)

        with open(path, 'w') as fp:
            yaml.dump(_config, fp)
        self.results = []
        self.stabe_results = []

    def reset(self):
        self.results = []
        self.stabe_results = []

    def append(self, result, algo, stable_result=None):
        self.results.append(result)
        # self.stabe_results.append(stable_result)
        self.save_algo(algo)
        # self.save_stabe()

    def append_compare_parent(self, result, no_par, stable_result=None):
        self.results.append(result)
        # self.stabe_results.append(stable_result)
        self.save_compare_parents(no_par)
        # self.save_stabe()

    def save_algo(self, algo):
        path = os.path.join('result', self.instance,
                            algo, '%d.pkl' % self.seed)
        folder = os.path.join('result', self.instance, algo)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'wb') as fp:
            pickle.dump(self.results, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def save_compare_parents(self, no_par):
        path = os.path.join('result', 'parents', self.instance, '%d' % no_par,
                            '%d.pkl' % self.seed)
        folder = os.path.join('result', 'parents',
                              self.instance, '%d' % no_par)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'wb') as fp:
            pickle.dump(self.results, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self):
        path = os.path.join('result', self.instance, '%d.pkl' % self.seed)
        folder = os.path.join('result', self.instance)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'wb') as fp:
            pickle.dump(self.results, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def save_stabe(self):
        path = os.path.join('result', self.instance,
                            'attribute%d.fun' % self.seed)
        folder = os.path.join('result', self.instance)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'wb') as fp:
            pickle.dump(self.results, fp, protocol=pickle.HIGHEST_PROTOCOL)
