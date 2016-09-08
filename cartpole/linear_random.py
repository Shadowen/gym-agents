import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib stuff
plt.ioff()

# Gym stuff
env = gym.make('CartPole-v0')

# Tensorflow stuff
sess = tf.Session()

tf_input_state = tf.placeholder(tf.float32, shape=[1, 4])

tf_param_w_0 = tf.Variable(tf.random_uniform([4, 1], minval=--1, maxval=1))

tf_output_action = tf.matmul(tf_input_state, tf_param_w_0)


def run_episode(env, parameters=None, render=False):
    if parameters is None:
        parameters = np.random.random([4, 1]) * 2 - 1
    tf.initialize_all_variables()
    observation = env.reset()
    total_reward = 0
    for _ in xrange(200):
        if render:
            env.render()
        action = 0 if sess.run(tf_output_action,
                               feed_dict={tf_input_state: observation[np.newaxis], tf_param_w_0: parameters})[0][
                          0] < 0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward, parameters


eps_to_solve = []
for trial in range(1000):
    print("Trial {}".format(trial))
    best_reward = 0
    best_params = None
    for episode in range(1000):
        total_reward, params = run_episode(env)

        if total_reward > best_reward:
            best_reward = total_reward
            best_params = params
            if best_reward >= 200:
                break
    eps_to_solve.append(episode)

plt.hist(eps_to_solve, 50, alpha=0.75)
plt.xlabel('Episodes')
plt.ylabel('Frequency')
plt.title('Solving CartPole with Linear Regression with Random Search')
plt.savefig('cartpole_linear_random')
plt.show()
