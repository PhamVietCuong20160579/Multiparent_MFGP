from mtsoo import *
from gym_evaluate import *
from utils import Saver

# def callback(result):
#     pass

env_list = yaml.load(open('env_list.yaml').read())
config = load_config()


def main():
    for exp_id in range(config['repeat']):
        for e in env_list:
            names = env_list[e]
            envs = GymTaskSet(names['names'])
            config.update(names)
            instance = e

            saver = Saver(config, instance, exp_id)
            callback = None
            # callback = saver.append_compare_parent
            callback = saver.append
            print('Testing for enviroments set: {}'.format(e))
            print('Enviroments list: {}'.format(names['names']))

            print('[+] EA - %d/%d' % (exp_id, config['repeat']))
            cea(envs, config, callback)
            print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
            mfea(envs, config, callback)
            print('[+] MFEAII - %d/%d' % (exp_id, config['repeat']))
            mfeaii(envs, config, callback)

            # no_par = 3
            # print('[+] MFEAII_MULTIPARENT - %d/%d' %
            #       (exp_id, config['repeat']))
            # mfeaii_mgp(envs, config, no_par, callback)
            # mfeaii_mgp_2(envs, config, no_par, callback)

            # print('[+] MFEAII_MULTIPARENT second ver - %d/%d; COMPARE NUMBER OF PARENT' %
            #       (exp_id, config['repeat']))
            # mfeaii_mgp(envs, config, no_par, callback)

            # saver.save()
            # saver.save_compare_parents(0)


if __name__ == '__main__':
    main()
