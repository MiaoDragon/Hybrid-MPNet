from plan_general import *
from Model.GEM_end2end_model import End2EndMPNet
from utility import *
import argparse
import random
import numpy as np
import torch
import os
from data_loader import load_test_dataset
import plan_s2d
from gem_eval import *
def IsInCollision(x,obc):
    size = 5.0
    s=np.zeros(2,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    cf = True
    for j in range(0,2):
        if abs(obc[j] - s[j]) > size/2.0:
            # not in collision
            cf=False
            break
    return cf


# test steerTo
print('steerTo test...')
start = torch.tensor([0.,-5.])
end = torch.tensor([4.99,0.])
obc = np.array([0.,0.])  # define a box at origin of size 5
size = 5.
print(steerTo(start, end, obc, IsInCollision, step_sz=0.01))
# test feasibility check
print('feasibility test...')
path = [torch.tensor([0.,6.]), torch.tensor([0.,2.])]
print(feasibility_check(path, obc, IsInCollision, step_sz=0.01))
# test lvc
print('lvc...')
path = [torch.tensor([0.,6.]), torch.tensor([0.,5.]),torch.tensor([0.,4.]),torch.tensor([0.,3.]),torch.tensor([-3.,3.]),torch.tensor([-3.,-3.]),
        torch.tensor([0.,-6.])]
print(lvc(path, obc, IsInCollision, step_sz=0.01))
pc_low = [[obc[0]-size/2,obc[1]-size/2] for i in range(1400)]
pc_low = np.array(pc_low)

pc_high = [[obc[0]+size/2,obc[1]+size/2] for i in range(1400)]
pc_high = np.array(pc_high)
obs = np.random.uniform(low=pc_low, high=pc_high)
obs = torch.from_numpy(obs).float().flatten()
print(obs)

# test neural replan
def main(args):
    # set seed
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    if args.memory_type == 'res':
        mpNet = End2EndMPNet(args.mlp_input_size, args.output_size, 'deep', \
                    args.n_tasks, args.n_memories, args.memory_strength, args.grad_step)
    elif args.memory_type == 'rand':
        #mpNet = End2EndMPNet_rand(args.mlp_input_size, args.output_size, 'deep', \
        #            args.n_tasks, args.n_memories, args.memory_strength, args.grad_step)
        pass
    # load previously trained model if start epoch > 0
    model_path='cmpnet_epoch_%d.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)

    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
        mpNet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))
    # load train and test data
    print('loading...')
    seen_test_data = load_test_dataset(N=1, NP=100, s=10, sp=4000, folder=args.data_path)
    unseen_test_data = load_test_dataset(N=1, NP=100,s=10, sp=0, folder=args.data_path)
    # test
    # setup evaluation function
    if args.env_type == 's2d':
        IsInCollision = plan_s2d.IsInCollision
    elif args.env_type == 'c2d':
        pass
    # testing
    print('testing...')
    seen_test_suc_rate = 0.
    unseen_test_suc_rate = 0.
    T = 1
    for _ in range(T):
        # unnormalize function
        unnormalize_func=lambda x: unnormalize(x, args.world_size)
        # seen
        time_file = os.path.join(args.model_path,'time_seen_epoch_%d_mlp.p' % (args.start_epoch))
        fes_path_, valid_path_ = eval_tasks(mpNet, seen_test_data, time_file, IsInCollision, unnormalize_func)
        valid_path = valid_path_.flatten()
        fes_path = fes_path_.flatten()   # notice different environments are involved
        # unseen
        time_file = os.path.join(args.model_path,'time_unseen_epoch_%d_mlp.p' % (args.start_epoch))
        fes_path_, valid_path_ = eval_tasks(mpNet, unseen_test_data, time_file, IsInCollision, unnormalize_func)
        valid_path = valid_path_.flatten()
        fes_path = fes_path_.flatten()   # notice different environments are involved

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
parser.add_argument('--world_size', type=int, default=50, help='boundary of world')
args = parser.parse_args()
print(args)
main(args)
