#!/usr/bin/env python3.7.7

import argparse
import datetime
import os
import random
import time
from threading import Thread
import numpy as np
import tensorflow as tf

from dqn import DeepQNetwork
import replay
from car_env import CarEnv
from state import State

parser = argparse.ArgumentParser()
parser.add_argument("--episode-timeout", type=int, default=60,
                    help="maximum episode amount of time allowed is seconds")
parser.add_argument("--train-epoch-steps", type=int, default=5000,
                    help="how many steps (=X frames) to run during a training epoch (approx -- will finish current game)")
parser.add_argument("--eval-epoch-steps", type=int, default=500,
                    help="how many steps (=X frames) to run during an eval epoch (approx -- will finish current game)")
parser.add_argument("--replay-capacity", type=int, default=100000, help="how many states to store for future training")
parser.add_argument("--prioritized-replay", action='store_true',
                    help="Prioritize interesting states when training (e.g. terminal or non zero rewards)")
parser.add_argument("--compress-replay", action='store_true',
                    help="if set replay memory will be compressed with blosc, allowing much larger replay capacity")
parser.add_argument("--normalize-weights", action='store_true',
                    help="if set weights/biases are normalized like torch, with std scaled by fan in to the node")
parser.add_argument("--save-model-freq", type=int, default=2000, help="save the model once per X training sessions")
parser.add_argument("--observation-steps", type=int, default=350, help="train only after this many step (=X frames)")
parser.add_argument("--learning-rate", type=float, default=0.0004,
                    help="learning rate (step size for optimization algo)")
parser.add_argument("--gamma", type=float, default=0.996,
                    help="gamma [0, 1] is the discount factor. It determines the importance of future rewards. A factor of 0 will make the agent consider only immediate reward, a factor approaching 1 will make it strive for a long-term high reward")
parser.add_argument("--target-model-update-freq", type=int, default=300,
                    help="how often (in steps) to update the target model.  Note nature paper says this is in 'number of parameter updates' but their code says steps. see tinyurl.com/hokp4y8")
parser.add_argument("--model", help="tensorflow model checkpoint file to initialize from")
parser.add_argument("--image-width", type=int, default=84, help="the width of the image")
parser.add_argument("--image-height", type=int, default=84, help="the height of the image")
parser.add_argument("--history-length", type=int, default=2, help="(>=1) length of history used in the dqn. An action is performed [history-length] time")
parser.add_argument("--epsilon", type=float, default=1, help="]0, 1]for epsilon greedy train")
parser.add_argument("--epsilon-decay", type=float, default=0.999995,
                    help="]0, 1] every step epsilon = epsilon * decay, in order to decrease constantly")
parser.add_argument("--epsilon-min", type=float, default=0.1, help="epsilon with decay doesn't fall below epsilon min")
parser.add_argument("--tensorboard-logging-freq", type=int, default=300,
                    help="save training statistics once every X steps")
parser.add_argument("--logging", type=bool, default=True, help="enable tensorboard logging")
args = parser.parse_args()

print('Arguments: ', args)

base_output_dir = 'run-out-' + time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(base_output_dir)

tensorboard_dir = base_output_dir + "/tensorboard/"
os.makedirs(tensorboard_dir)
summary_writer = tf.summary.create_file_writer(tensorboard_dir)
with summary_writer.as_default():
    tf.summary.text('params', str(args), step=0)

State.setup(args)

environment = CarEnv(args)
replayMemory = replay.ReplayMemory(base_output_dir, args)
dqn = DeepQNetwork(environment.get_num_actions(), environment.get_state_size(),
                   replayMemory, base_output_dir, tensorboard_dir, args)

train_epsilon = args.epsilon  # don't want to reset epsilon between epoch
start_time = datetime.datetime.now()
train_episodes = 0
eval_episodes = 0
episode_train_reward_list = []
episode_eval_reward_list = []

stop = False


def stop_handler():
    global stop
    while not stop:
        user_input = input()
        if user_input == 'q':
            print("Stopping...")
            stop = True


process = Thread(target=stop_handler)
process.start()

episode_min_time = 0.015  # minimum time required per step execution

train_epsilon = args.epsilon  # don't want to reset epsilon between epoch
startTime = datetime.datetime.now()


def run_epoch(minEpochSteps, evalWithEpsilon=None):
    global train_epsilon
    global train_episodes
    global eval_episodes
    global episode_train_reward_list
    global episode_eval_reward_list
    global train_epsilon
    stepStart = environment.getStepNumber()
    is_training = True if evalWithEpsilon is None else False
    startGameNumber = environment.get_game_number()
    epochTotalScore = 0
    step_time_mean = 0.0

    while environment.getStepNumber() - stepStart < minEpochSteps and not stop:
        stateReward = 0
        state = None

        episode_losses = []

        epStartTime = datetime.datetime.now()

        while not environment.isGameOver() and not stop:
            step_time_start = datetime.datetime.now()

            # Choose next action
            if evalWithEpsilon is None:
                epsilon = train_epsilon
            else:
                epsilon = evalWithEpsilon

            if train_epsilon > args.epsilon_min:
                train_epsilon = train_epsilon * args.epsilon_decay
                if train_epsilon < args.epsilon_min:
                    train_epsilon = args.epsilon_min

            if state is None or random.random() < epsilon:
                action = random.randrange(environment.get_num_actions())  # random action
            else:
                screens = np.reshape(state.get_screens(), (1, State.IMAGE_HEIGHT, State.IMAGE_WIDHT, args.history_length))
                action = dqn.inference(screens)  # this one takes the decision based on input

            # Make the move
            oldState = state
            reward, state, isTerminal = environment.step(action)

            # Record experience in replay memory and train
            if is_training and oldState is not None:
                clippedReward = min(1, max(-1, reward))
                replayMemory.add_sample(replay.Sample(oldState, action, clippedReward, state, isTerminal))

                if environment.getStepNumber() > args.observation_steps and environment.getEpisodeStepNumber() % args.history_length == 0:
                    batch = replayMemory.draw_batch(32)
                    loss = dqn.train(batch, environment.getStepNumber())
                    episode_losses.append(loss)

            if isTerminal:
                state = None

            # check if episode time reaches timeout max time
            epStopTime = datetime.datetime.now() - epStartTime
            if epStopTime.total_seconds() > args.episode_timeout:
                break

            # calculate step time and mean
            step_time_stop = datetime.datetime.now()
            step_delta = (step_time_stop - step_time_start).total_seconds()
            if step_time_mean < episode_min_time:
                step_time_mean = step_delta
            else:
                step_time_mean = step_time_mean * 0.999 + step_delta * 0.001
            step_time_mean = max(step_time_mean, episode_min_time)
            if step_delta < step_time_mean:
                time.sleep(step_time_mean - step_delta)  # wait for mean step time to be reached

        #################################
        # logging
        #################################

        episode_time = datetime.datetime.now() - startTime

        if is_training:
            train_episodes += 1
            episode_train_reward_list.insert(0, environment.getGameScore())
            if len(episode_train_reward_list) > 100:
                episode_train_reward_list = episode_train_reward_list[:-1]
            avg_rewards = np.mean(episode_train_reward_list)

            episode_avg_loss = 0
            if episode_losses:
                episode_avg_loss = np.mean(episode_losses)

            log = ('Episode %d ended with score: %.2f (%s elapsed) (step: %d). Avg score: %.2f Avg loss: %.5f' %
                   (environment.get_game_number(), environment.getGameScore(), str(episode_time),
                    environment.getStepNumber(), avg_rewards, episode_avg_loss))
            print(log)
            print("   epsilon " + str(train_epsilon))
            if args.logging:
                with summary_writer.as_default():
                    tf.summary.scalar('train episode reward', environment.getGameScore(), step=train_episodes)
                    tf.summary.scalar('train avg reward(100)', avg_rewards, step=train_episodes)
                    tf.summary.scalar('average loss', episode_avg_loss, step=train_episodes)
                    tf.summary.scalar('epsilon', train_epsilon, step=train_episodes)
                    tf.summary.scalar('steps', environment.getStepNumber(), step=train_episodes)
                    tf.summary.scalar('step time (mean, ms)', step_time_mean*1000, step=train_episodes)
        else:
            eval_episodes += 1
            episode_eval_reward_list.insert(0, environment.getGameScore())
            if len(episode_eval_reward_list) > 100:
                episode_eval_reward_list = episode_eval_reward_list[:-1]
            avg_rewards = np.mean(episode_eval_reward_list)

            log = ('Eval %d ended with score: %.2f (%s elapsed) (step: %d). Avg score: %.2f' %
                   (environment.get_game_number(), environment.getGameScore(), str(episode_time),
                    environment.getStepNumber(), avg_rewards))
            print(log)
            if args.logging:
                with summary_writer.as_default():
                    tf.summary.scalar('eval episode reward', environment.getGameScore(), step=eval_episodes)
                    tf.summary.scalar('eval avg reward(100)', avg_rewards, step=eval_episodes)

        print('   Step time mean = %.3f --> %dFPS' % (step_time_mean, int(1 / step_time_mean)))

        epochTotalScore += environment.getGameScore()
        environment.resetGame()

    # return the average score
    if environment.get_game_number() - startGameNumber == 0:
        return 0
    return epochTotalScore / (environment.get_game_number() - startGameNumber)


while not stop:
    aveScore = run_epoch(args.train_epoch_steps)  # train
    print('Average training score: %d' % aveScore)
    print('\a')
    aveScore = run_epoch(args.eval_epoch_steps, evalWithEpsilon=.0)  # eval
    print('Average eval score: %d' % aveScore)
    print('\a')

environment.stop()
