from mtsoo import *
from function_evaluate import *
from scipy.io import savemat


def callback(res):
    pass


def main():
    config = load_config()
    functions = CI_HS().functions

    for exp_id in range(config['repeat']):
        print('[+] EA - %d/%d' % (exp_id, config['repeat']))
        cea(functions, config, callback)
        print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
        mfea(functions, config, callback)
        print('[+] MFEAII - %d/%d' % (exp_id, config['repeat']))
        mfeaii(functions, config, callback)

        # print('[+] MFEAII_MULTIPARENT - %d/%d' % (exp_id, config['repeat']))
        # mfeaii_mgp(functions, config, callback, normal_beta=False)

        # print('[+] MFEAII_MULTIPARENT (beta calculated from distribution model) - %d/%d' %
        #       (exp_id, config['repeat']))
        # mfeaii_mgp(functions, config, callback, normal_beta=True)


if __name__ == '__main__':
    main()
