# Custom gym enviroments

These are some custom gym enviroments based on open AI gym enviroments, with the purpose of testing multitasking / multi-objective / multi-factorial evolution code such as MFEA, MFEAII, MO-MFEAII on reinforcement learning enviroments

## How to use:

install: using pip

        pip install -e  .

how to use: import with gym as normally, then call make with prefix "custom_gym:" and name of enviroments.

        import gym
        env = gym.make("custom_gym:cartpole-swing-up-v0")

## Note:

currently, cartpole-swing-up-v0 require extra input, therefore the input is different from normal cartpole
