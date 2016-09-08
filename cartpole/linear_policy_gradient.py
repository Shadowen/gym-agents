# Based on http://kvfrans.com/simple-algoritms-for-solving-cartpole/
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util.realtime_plotter import RealtimePlotter

PLOTTER_ON = True


def create_policy_estimator():
    with tf.variable_scope("policy"):
        # Prediction
        tf_input_state = tf.placeholder(tf.float32, [None, 4])
        params = tf.Variable(tf.random_uniform([4, 2], minval=-1, maxval=1), dtype=tf.float32)
        tf_output_action_probabilities = tf.nn.softmax(tf.matmul(tf_input_state, params))
        # Training
        tf_input_action_targets = tf.placeholder(tf.float32, [None, 2])
        cross_entropy = tf.reduce_sum(tf.mul(tf_input_action_targets, tf.log(tf_output_action_probabilities)),
                                      reduction_indices=[1])
        tf_input_advantages = tf.placeholder(tf.float32, [None, 1])
        cost_function = -tf.reduce_sum(cross_entropy * tf_input_advantages)
        tf_optimizer = tf.train.AdamOptimizer(0.1).minimize(cost_function)
        return (
            tf_input_state, tf_input_action_targets, tf_input_advantages, tf_output_action_probabilities, tf_optimizer)


def create_value_estimator():
    with tf.variable_scope("value"):
        # Prediction
        tf_input_state = tf.placeholder("float", [None, 4])
        w1 = tf.Variable(tf.random_uniform([4, 10], minval=-1, maxval=1), dtype=tf.float32)
        b1 = tf.Variable(tf.random_uniform([10], minval=-1, maxval=1), dtype=tf.float32)
        h1 = tf.nn.sigmoid(tf.matmul(tf_input_state, w1) + b1)
        w2 = tf.Variable(tf.random_uniform([10, 1], minval=-1, maxval=1), dtype=tf.float32)
        b2 = tf.Variable(tf.random_uniform([1], minval=-1, maxval=1), dtype=tf.float32)
        tf_output_predicted_values = tf.matmul(h1, w2) + b2
        # Training
        tf_input_target_values = tf.placeholder("float", [None, 1])
        cost_function = tf.nn.l2_loss(tf_output_predicted_values - tf_input_target_values)
        tf_optimizer = tf.train.AdamOptimizer(0.1).minimize(cost_function)
        return tf_input_state, tf_input_target_values, tf_output_predicted_values, tf_optimizer


def run_episode(env, policy_estimator, value_estimator, sess, render=False):
    pl_state, target_actions, pl_advantages, pl_calculated, pl_optimizer = policy_estimator
    vl_state, target_values, vl_calculated, vl_optimizer = value_estimator
    observation = env.reset()
    total_reward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    # Sample
    for _ in range(200):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})
        action = 0 if np.random.uniform(0, 1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        actions.append(np.array([1 if i == action else 0 for i in range(2)]))
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        if render: env.render()
        transitions.append((old_observation, action, reward))
        total_reward += reward

        if done:
            break

    # Learn
    discounted_future_reward = 0
    transitions.reverse()
    for trans in transitions:
        obs, action, reward = trans

        # Calculate return from now to end (Monte-Carlo)
        discounted_future_reward = reward + discounted_future_reward * 0.97
        value_estimate = sess.run(vl_calculated, feed_dict={vl_state: np.expand_dims(obs, axis=0)})[0][0]

        update_vals.append(discounted_future_reward)
        advantages.append(discounted_future_reward - value_estimate)

    # Update
    sess.run(vl_optimizer, feed_dict={vl_state: states, target_values: np.expand_dims(update_vals, axis=1)})
    sess.run(pl_optimizer,
             feed_dict={pl_state: states, target_actions: actions, pl_advantages: np.expand_dims(advantages, axis=1)})

    return total_reward


if PLOTTER_ON:
    plotter = RealtimePlotter()
eps_to_solve = []
for trial in range(100):
    env = gym.make('CartPole-v0')
    tf.reset_default_graph()
    policy_estimator = create_policy_estimator()
    value_estimator = create_value_estimator()
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    if PLOTTER_ON:
        plotter.new_line()

    rewards = []
    for episodes in range(2000):
        reward = run_episode(env, policy_estimator, value_estimator, sess)
        rewards.append(reward)
        if PLOTTER_ON:
            plotter.update(rewards)
        if reward >= 200:
            break

    eps_to_solve.append(episodes)

plt.figure()
plt.hist(eps_to_solve, 50, alpha=0.75)
plt.xlabel('Episodes')
plt.ylabel('Frequency')
plt.title('Solving CartPole with Linear Policy Gradient')
plt.savefig('cartpole_linear_policy_gradient_sigmoid_cross_entropy')
plt.show()
