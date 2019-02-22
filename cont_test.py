'''
This is the main file to run gem_end2end network.
It simulates the real scenario of observing a data, puts it inside the memory (or not),
and trains the network using the data
after training at each step, it will output the R matrix described in the paper
https://arxiv.org/abs/1706.08840
and after sevral training steps, it needs to store the parameter in case emergency
happens
To make it work in a real-world scenario, it needs to listen to the observer at anytime,
and call the network to train if a new data is available
(this thus needs to use multi-process)
here for simplicity, we just use single-process to simulate this scenario
'''
from __future__ import print_function
from GEM_end2end_model import End2EndMPNet
from GEM_end2end_model_rand import End2EndMPNet as End2EndMPNet_rand
#from gem_observer import Observer
import numpy as np
import argparse
import os
import torch
from gem_eval_simple import eval_tasks as eval_tasks_s2d
from gem_eval_complex import eval_tasks as eval_tasks_c2d
from data_loader import load_dataset, load_test_dataset
from torch.autograd import Variable
import copy
import os
import gc
import random
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def save_state(net, fname):
    # save both model state and optimizer state
    states = {
        'state_dict': net.state_dict(),
        'optimizer': net.opt.state_dict()
    }
    torch.save(states, fname)

def load_net_state(net, fname):
    checkpoint = torch.load(fname)
    net.load_state_dict(checkpoint['state_dict'])
def load_opt_state(net, fname):
    checkpoint = torch.load(fname)
    net.opt.load_state_dict(checkpoint['optimizer'])


def main(args):
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    if args.memory_type == 'res':
        mpNet = End2EndMPNet(args.mlp_input_size, args.output_size, 'deep', \
                    args.n_tasks, args.n_memories, args.memory_strength, args.grad_step)
    elif args.memory_type == 'rand':
        mpNet = End2EndMPNet_rand(args.mlp_input_size, args.output_size, 'deep', \
                    args.n_tasks, args.n_memories, args.memory_strength, args.grad_step)
    # load previously trained model if start epoch > 0
    model_path='mpNet_cont_train_epoch_%d.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
        mpNet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))
    # load train and test data
    print('loading...')
    seen_test_data = load_test_dataset(N=100, NP=200, s=0, sp=4000, folder=args.data_path)
    unseen_test_data = load_test_dataset(N=10, NP=2000,s=100, sp=0, folder=args.data_path)
    # test
    # setup evaluation function
    if args.env_type == 's2d':
        eval_tasks = eval_tasks_s2d
    elif args.env_type == 'c2d':
        eval_tasks = eval_tasks_c2d
    # testing
    print('testing...')
    seen_test_suc_rate = 0.
    unseen_test_suc_rate = 0.
    T = 1
    for _ in range(T):
        # seen
        time_file = os.path.join(args.model_path,'time_seen_epoch_%d_mlp.p' % (args.start_epoch))
        fes_path_, valid_path_ = eval_tasks(mpNet, seen_test_data, time_file)
        valid_path = valid_path_.flatten()
        fes_path = fes_path_.flatten()   # notice different environments are involved
        seen_test_suc_rate += fes_path.sum() / valid_path.sum()
        # unseen
        time_file = os.path.join(args.model_path,'time_unseen_epoch_%d_mlp.p' % (args.start_epoch))
        fes_path_, valid_path_ = eval_tasks(mpNet, unseen_test_data, time_file)
        valid_path = valid_path_.flatten()
        fes_path = fes_path_.flatten()   # notice different environments are involved
        unseen_test_suc_rate += fes_path.sum() / valid_path.sum()
    seen_test_suc_rate = seen_test_suc_rate / T
    unseen_test_suc_rate = unseen_test_suc_rate / T    # Save the models
    f = open(os.path.join(args.model_path,'seen_accuracy_epoch_%d.txt' % (args.start_epoch)), 'w')
    f.write(str(seen_test_suc_rate))
    f.close()
    f = open(os.path.join(args.model_path,'unseen_accuracy_epoch_%d.txt' % (args.start_epoch)), 'w')
    f.write(str(unseen_test_suc_rate))
    f.close()


parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
parser.add_argument('--no_env', type=int, default=100,help='directory for obstacle images')
parser.add_argument('--no_motion_paths', type=int,default=4000,help='number of optimal paths in each environment')
parser.add_argument('--grad_step', type=int, default=1, help='number of gradient steps in continual learning')
# for continual learning
parser.add_argument('--n_tasks', type=int, default=1,help='number of tasks')
parser.add_argument('--n_memories', type=int, default=256, help='number of memories for each task')
parser.add_argument('--memory_strength', type=float, default=0.5, help='memory strength (meaning depends on memory)')
# Model parameters
parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--device', type=int, default=0, help='cuda device')
parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--memory_type', type=str, default='res', help='res for reservoid, rand for random sampling')
parser.add_argument('--env_type', type=str, default='s2d', help='s2d for simple 2d, c2d for complex 2d')
args = parser.parse_args()
print(args)
main(args)
