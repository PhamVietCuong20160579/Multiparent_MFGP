from mtsoo import *
from gym_evaluate import *
from utils import Saver

instance = 'Gym-CartPole-variation'


# def callback(result):
#     pass

env_list = yaml.load(open('env_list.yaml').read())
config = load_config()


def main():
    for exp_id in range(config['repeat']):
        # functions = CI_HS().functions
        for e in env_list:
            names = env_list[e]
            envs = GymTaskSet(names['names'])
            config.update(names)
            instance = e

            saver = Saver(config, instance, exp_id)
            callback = saver.append
            print('Testing for enviroments set: {}'.format(e))
            print('Enviroments list: {}'.format(names['names']))

            print('[+] EA - %d/%d' % (exp_id, config['repeat']))
            cea_re = cea(envs, config, callback)
            print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
            mfea_re = mfea(envs, config, callback)
            print('[+] MFEAII - %d/%d' % (exp_id, config['repeat']))
            mfeaii_re = mfeaii(envs, config, callback)

            print('[+] MFEAII_MULTIPARENT - %d/%d' %
                  (exp_id, config['repeat']))
            mfeaii_mgp_re = mfeaii_mgp(envs, config, callback)

            saver.save()


if __name__ == '__main__':
    main()
