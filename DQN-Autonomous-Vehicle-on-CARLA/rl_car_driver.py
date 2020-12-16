#!/usr/bin/env python3.7.7

###############################################################################
# c-ray - a simple raytracer
#
# http://www.futuretech.blinkenlights.nl/c-ray.html
#
# Copyright (C) 2006 John Tsiombikas <nuclear@siggraph.org>
# Copyright (C) 2016, 2017, 2018 Moreno Marzolla <moreno.marzolla@unibo.it>
#
# You are free to use, modify and redistribute this program under the
# terms of the GNU General Public License v2 or (at your option) later.
# see "http://www.gnu.org/licenses/gpl.txt" for details.
# ---------------------------------------------------------------------------
# Usage:
#
###############################################################################

#
from threading import Thread
import sys
import numpy as np
import os
import random
import replay
import time
import argparse
import datetime

import dqn
from car_env import CarEnv
from state import State

parser = argparse.ArgumentParser()
parser.add_argument("--train-epoch-steps", type=int, default=5000, help="how many steps (=X frames) to run during a training epoch (approx -- will finish current game)")
parser.add_argument("--eval-epoch-steps", type=int, default=500, help="how many steps (=X frames) to run during an eval epoch (approx -- will finish current game)")
parser.add_argument("--replay-capacity", type=int, default=100000, help="how many states to store for future training")
parser.add_argument("--prioritized-replay", action='store_true', help="Prioritize interesting states when training (e.g. terminal or non zero rewards)")
parser.add_argument("--compress-replay", action='store_true', help="if set replay memory will be compressed with blosc, allowing much larger replay capacity")
parser.add_argument("--normalize-weights", action='store_true', help="if set weights/biases are normalized like torch, with std scaled by fan in to the node")
parser.add_argument("--save-model-freq", type=int, default=2000, help="save the model once per X training sessions")
parser.add_argument("--observation-steps", type=int, default=350, help="train only after this many stesp (=X frames)")
parser.add_argument("--learning-rate", type=float, default=0.0004, help="learning rate (step size for optimization algo)")
parser.add_argument("--gamma", type=float, default=0.996, help="gamma [0, 1] is the discount factor. It determines the importance of future rewards. A factor of 0 will make the agent consider only immediate reward, a factor approaching 1 will make it strive for a long-term high reward")
parser.add_argument("--target-model-update-freq", type=int, default=300, help="how often (in steps) to update the target model.  Note nature paper says this is in 'number of parameter updates' but their code says steps. see tinyurl.com/hokp4y8")
parser.add_argument("--model", help="tensorflow model checkpoint file to initialize from")
parser.add_argument("--frame", type=int, default=2, help="frame per step")
parser.add_argument("--epsilon", type=float, default=1, help="]0, 1]for epsilon greedy train")
parser.add_argument("--epsilon-decay", type=float, default=0.99998, help="]0, 1] every step epsilon = epsilon * decay, in order to decrease constantly")
parser.add_argument("--epsilon-min", type=float, default=0.1, help="epsilon with decay doesn't fall below epsilon min")
parser.add_argument("--tensorboard-logging-freq", type=int, default=300, help="save training statistics once every X steps")
args = parser.parse_args()

print('Arguments: ', (args))

baseOutputDir = 'run-out-' + time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(baseOutputDir)

State.setup(args)

environment = CarEnv(args, baseOutputDir)
dqn = dqn.DeepQNetwork(environment.getNumActions(), baseOutputDir, args)
replayMemory = replay.ReplayMemory(args)

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

train_epsilon = args.epsilon #don't want to reset epsilon between epoch
startTime = datetime.datetime.now()

def runEpoch(minEpochSteps, evalWithEpsilon=None):
    global train_epsilon
    stepStart = environment.getStepNumber()
    isTraining = True if evalWithEpsilon is None else False
    startGameNumber = environment.getGameNumber()
    epochTotalScore = 0

    while environment.getStepNumber() - stepStart < minEpochSteps and not stop:
        stateReward = 0
        state = None

        epStartTime = datetime.datetime.now()
        
        while not environment.isGameOver() and not stop:
            # Choose next action
            if evalWithEpsilon is None:
                epsilon = train_epsilon
            else:
                epsilon = evalWithEpsilon

            if train_epsilon > args.epsilon_min:
                train_epsilon = train_epsilon * args.epsilon_decay
                if train_epsilon < args.epsilon_min:
                    train_epsilon = args.epsilon_min

            if state is None or random.random() < (epsilon):
                action = random.randrange(environment.getNumActions())
            else:
                screens = np.reshape(state.getScreens(), (1, State.IMAGE_SIZE, State.IMAGE_SIZE, args.frame))
                action = dqn.inference(screens)

            # Make the move
            oldState = state
            reward, state, isTerminal = environment.step(action)
            
            # Record experience in replay memory and train
            if isTraining and oldState is not None:
                clippedReward = min(1, max(-1, reward))
                replayMemory.addSample(replay.Sample(oldState, action, clippedReward, state, isTerminal))

                if environment.getStepNumber() > args.observation_steps and environment.getEpisodeStepNumber() % args.frame == 0:
                    batch = replayMemory.drawBatch(32)
                    dqn.train(batch, environment.getStepNumber())

            if isTerminal:
                state = None

            epStopTime = datetime.datetime.now() - epStartTime
            if epStopTime.total_seconds() > 60:
                break

        episodeTime = datetime.datetime.now() - startTime
        print('%s %d ended with score: %d (%s elapsed)' %
            ('Episode' if isTraining else 'Eval', environment.getGameNumber(), environment.getGameScore(), str(episodeTime)))
        if isTraining:
          print("epsilon " + str(train_epsilon))
        epochTotalScore += environment.getGameScore()
        environment.resetGame()
    
    # return the average score
    if environment.getGameNumber() - startGameNumber == 0:
        return 0
    return epochTotalScore / (environment.getGameNumber() - startGameNumber)


while not stop:
    aveScore = runEpoch(args.train_epoch_steps) #train
    print('Average training score: %d' % (aveScore))
    print('\a')
    aveScore = runEpoch(args.eval_epoch_steps, evalWithEpsilon=.0) #eval
    print('Average eval score: %d' % (aveScore))
    print('\a')

environment.stop()