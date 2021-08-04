"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from src.game import Game
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=40, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=300000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch number")
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--render", type=bool, default=False)

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env = Game(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    # pre_model = torch.load("{}/tetris".format(opt.saved_path),  map_location=torch.device('cpu'))
    # model.load_state_dict(pre_model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    start_epoch = opt.start_epoch
    if opt.resume:
        if os.path.isfile("{}/checkpoint".format(opt.saved_path)):
            checkpoint = torch.load("{}/checkpoint".format(opt.saved_path))
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    state = env.start()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = start_epoch
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action)
        # print(action, reward)
        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.tetris.score
            final_tetrominoes = env.tetris.tetrominoes
            final_cleared_lines = env.tetris.cleared_lines
            # print(final_score, final_tetrominoes, final_cleared_lines)
            print(','.join(env.tetris.op_record))
            state = env.start()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()
        if final_score > 0:
            print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
                epoch,
                opt.num_epochs,
                action,
                final_score,
                final_tetrominoes,
                final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            if not os.path.isdir(opt.saved_path):
                  os.mkdir(opt.saved_path)
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, "{}/checkpoint_{}".format(opt.saved_path, epoch))
            torch.save(checkpoint, "{}/checkpoint".format(opt.saved_path))

    torch.save(model, "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
