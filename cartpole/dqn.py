# Based on http://kvfrans.com/simple-algoritms-for-solving-cartpole/
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util.realtime_plotter import RealtimePlotter

PLOTTER_ON = True


class QEstimator:
    def __init__(self, scope, weights=None):
        with tf.variable_scope(scope):
            # Prediction
            self.input_state = tf.placeholder(tf.float32, [None, 4], name='input_state')
            self.weights.w1 = tf.Variable(tf.random_uniform([4, 10], minval=-1, maxval=1),
                                          dtype=tf.float32) if weights is None else weights.w1
            self.weights.b1 = tf.Variable(tf.random_uniform([10], minval=-1, maxval=1),
                                          dtype=tf.float32) if weights is None else weights.b1
            self.h1 = tf.nn.relu(tf.matmul(self.input_state, self.weights.w1) + self.weights.b1)
            self.weights.w2 = tf.Variable(tf.random_uniform([10, 2], minval=-1, maxval=1),
                                          dtype=tf.float32) if weights is None else weights.w2
            self.weights.b2 = tf.Variable(tf.random_uniform([2], minval=-1, maxval=1),
                                          dtype=tf.float32) if weights is None else weights.b2
            self.output_predicted_q = tf.matmul(self.h1, self.weights.w2) + self.weights.b2
            self.best_action = tf.argmax(self.output_predicted_q, dimension=1)

    def get_action(self, sess, state):
        return sess.run(self.best_action, feed_dict={self.input_state: state})


class Agent:
    def __init__(self, env, alpha=tf.constant(0.1, name='learning_rate'), gamma=tf.constant(0.99, name='gamma')):
        self.env = env

        # Prediction
        self.estimator = QEstimator("estimator")
        # Training
        self.input_reward = tf.placeholder(tf.float32, [None, 1], name='input_reward')
        self.input_target_actions = tf.placeholder(tf.float32, [None, 2], name='input_target_actions')
        self.next_estimator = QEstimator("next_estimator", self.estimator.weights)
        target_q = tf.stop_gradient(
            self.input_reward + gamma * tf.reduce_max(self.next_estimator.output_predicted_q, reduction_indices=[1],
                                                      keep_dims=True),
            name='target_q')
        self.loss_function = tf.nn.l2_loss((target_q - self.estimator.output_predicted_q),
                                           name='loss_function')
        self.optimizer = tf.train.AdamOptimizer(alpha).minimize(self.loss_function)

    def get_eps_greedy_action(self, sess, state, epsilon=0.3):
        return self.estimator.get_action(sess, state)[0] if np.random.random() > epsilon else env.action_space.sample()


class ReplayMemory:
    def __init__(self, max_size=1000):
        self.max_length = max_size
        self.length = 0
        self.states = np.empty([max_size, 4])
        self.actions = np.empty([max_size, 2])
        self.rewards = np.empty([max_size, 1])
        self.next_states = np.empty([max_size, 4])

    def add_transition(self, state, action, reward, next_state):
        # TODO
        self.states = np.roll(self.states, 1, 0)
        self.states[0] = state
        self.actions = np.roll(self.actions, 1, 0)
        self.actions[0] = action
        self.rewards = np.roll(self.rewards, 1, 0)
        self.rewards[0] = reward
        self.next_states = np.roll(self.next_states, 1, 0)
        self.next_states[0] = next_state
        if self.length < self.max_length:
            self.length += 1

    def random_sample(self, num_samples=1):
        batch_indices = np.random.choice(len(self), num_samples)
        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        next_states = self.next_states[batch_indices]
        return states, actions, rewards, next_states

    def get_all(self):
        states = self.states[:len(self)]
        actions = self.actions[:len(self)]
        rewards = self.rewards[:len(self)]
        next_states = self.next_states[:len(self)]
        return states, actions, rewards, next_states

    def __len__(self):
        return self.length


def action_vector_to_onehot(action, length):
    return [1 if i == action else 0 for i in range(length)]


def do_train(sess, q_optimizer, replay_memory, batch_size=10):
    states, actions, rewards, next_states = replay_memory.random_sample()
    sess.run(q_optimizer.optimizer,
             feed_dict={q_optimizer.estimator.input_state: states, q_optimizer.input_target_actions: actions,
                        q_optimizer.input_reward: rewards, q_optimizer.next_estimator.input_state: next_states})


def run_episode(env, q_optimizer, sess, replay_memory, render=False):
    observation = env.reset()
    next_state = np.expand_dims(observation, axis=0)

    # Rollout
    total_reward = 0
    for _ in range(200):
        state = next_state
        action = q_optimizer.get_eps_greedy_action(sess, state)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        next_state = np.expand_dims(observation, axis=0)
        # Store transition in replay memory
        replay_memory.add_transition(state[0], action_vector_to_onehot(action, 2), reward, next_state[0])
        # Training
        do_train(sess, q_optimizer, replay_memory)
        if done:
            break
    return total_reward


if PLOTTER_ON:
    plotter = RealtimePlotter()

env = gym.make('CartPole-v0')
q_optimizer = Agent(env)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter('./tensorboard', sess.graph)
if PLOTTER_ON:
    plotter.new_line()

replay_memory = ReplayMemory()
rewards_plot = []
losses = []
for episode in range(2000):
    print("Episode {}".format(episode))
    total_reward = run_episode(env, q_optimizer, sess, replay_memory)
    rewards_plot.append(total_reward)
    states, actions, rewards, next_states = replay_memory.get_all()
    losses.append(
        sess.run(q_optimizer.loss_function, feed_dict={q_optimizer.estimator.input_state: states,
                                                       q_optimizer.input_target_actions: actions,
                                                       q_optimizer.input_reward: rewards,
                                                       q_optimizer.next_estimator.input_state: next_states}) / len(
            replay_memory))
    if PLOTTER_ON:
        plotter.update(rewards_plot)
    if total_reward >= 200:
        break
#
# eps_to_solve.append(episodes)
#
# plt.figure()
# plt.hist(eps_to_solve, 50, alpha=0.75)
# plt.xlabel('Episodes')
# plt.ylabel('Frequency')
# plt.title('Solving CartPole with SARSA(lambda)')
# plt.savefig('cartpole_sarsa_lambda')
# plt.show()
