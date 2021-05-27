from mtsoo import *
from gym_evaluate import *
from utils import Saver

instance = 'Gym-CartPole-variation'


# def callback(result):
#     pass


def main():
    config = load_config()
    # functions = CI_HS().functions
    envs = GymTaskSet(config['names'])

    for exp_id in range(config['repeat']):
        saver = Saver(config, instance, exp_id)
        callback = saver.append

        # print('[+] EA - %d/%d' % (exp_id, config['repeat']))
        # cea(envs, config, callback)
        # print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
        # mfea(envs, config, callback)
        # print('[+] MFEAII - %d/%d' % (exp_id, config['repeat']))
        # mfeaii(envs, config, callback)

        print('[+] MFEAII_MULTIPARENT - %d/%d' % (exp_id, config['repeat']))
        mfeaii_mgp_2(envs, config, callback, normal_beta=False)

        # print('[+] MFEAII_MULTIPARENT (beta calculated from distribution model) - %d/%d' %
        #       (exp_id, config['repeat']))
        # mfeaii_mgp(envs, config, callback, normal_beta=True, const_rmp=True)

        saver.save()


if __name__ == '__main__':
    main()
