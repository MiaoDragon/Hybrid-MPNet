"""
read in generated paths. Plan path using neural planner.
For failed segments, use informed-rrt* to generate demonstration. When the segment is the
entire path, just use the preloaded data as demonstration.
Load the demonstration into replay memory and reservoid memory
"""
from __future__ import print_function
from GEM_end2end_model import End2EndMPNet
#from gem_observer import Observer
import numpy as np
import argparse
import os
import torch
from gem_eval_complex import eval_tasks
from data_loader import *
from torch.autograd import Variable
import copy
import os
import gc
import random
import time
from informed_rrtstar import RRT
DEFAULT_STEP = 0.05
mpNet = None
num_trained_samples = 0

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

def get_input(i,data,targets,bs):

    if i+bs<len(data):
        bi=data[i:i+bs]
        bt=targets[i:i+bs]
    else:
        bi=data[i:]
        bt=targets[i:]

    #return torch.from_numpy(bi),torch.from_numpy(bt)
    return torch.tensor(bi), torch.tensor(bt)

def load_net_state(net, fname):
    checkpoint = torch.load(fname)
    net.load_state_dict(checkpoint['state_dict'])
def load_opt_state(net, fname):
    checkpoint = torch.load(fname)
    net.opt.load_state_dict(checkpoint['optimizer'])

def steerTo(start, end, obc, step_sz=DEFAULT_STEP):
    DISCRETIZATION_STEP=step_sz
    dists=np.zeros(2,dtype=np.float32)
    for i in range(0,2):
        dists[i] = end[i] - start[i]
    distTotal = 0.0
    for i in range(0,2):
        distTotal =distTotal+ dists[i]*dists[i]
    distTotal = math.sqrt(distTotal)
    if distTotal>0:
        incrementTotal = distTotal/DISCRETIZATION_STEP
        for i in range(0,2):
            dists[i] =dists[i]/incrementTotal
        numSegments = int(math.floor(incrementTotal))
        stateCurr = np.zeros(2,dtype=np.float32)
        for i in range(0,2):
            stateCurr[i] = start[i]
        for i in range(0,numSegments):
            if IsInCollision(stateCurr,obc):
                return 0
            for j in range(0,2):
                stateCurr[j] = stateCurr[j]+dists[j]
        if IsInCollision(end,obc):
            return 0
    return 1
# checks the feasibility of entire path including the path edges
def feasibility_check(path,obc,step_sz=DEFAULT_STEP):
    for i in range(0,len(path)-1):
        ind=steerTo(path[i],path[i+1],obc,step_sz=step_sz)
        if ind==0:
            return 0
    return 1
# checks the feasibility of path nodes only
def collision_check(path,obc):
    for i in range(0,len(path)):
        if IsInCollision(path[i],obc):
            return 0
    return 1

def IsInCollision(x,obc):
    size = 5.0
    s=np.zeros(2,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    for i in range(0,7):
        cf=True
        for j in range(0,2):
            if abs(obc[i][j] - s[j]) > size/2.0:
                cf=False
                break
        if cf==True:
            return True
    return False
def is_reaching_target(start1,start2):
    s1=np.zeros(2,dtype=np.float32)
    s1[0]=start1[0]
    s1[1]=start1[1]
    s2=np.zeros(2,dtype=np.float32)
    s2[0]=start2[0]
    s2[1]=start2[1]
    for i in range(0,2):
        if abs(s1[i]-s2[i]) > 1.0:
            return False
    return True
#lazy vertex contraction
def lvc(path,obc,step_sz=DEFAULT_STEP):
    for i in range(0,len(path)-1):
        for j in range(len(path)-1,i+1,-1):
            ind=0
            ind=steerTo(path[i],path[j],obc,step_sz=step_sz)
            if ind==1:
                pc=[]
                for k in range(0,i+1):
                    pc.append(path[k])
                for k in range(j,len(path)):
                    pc.append(path[k])
                return lvc(pc,obc,step_sz=step_sz)
    return path

def transformToTrain(path, path_length, obs, obs_i):
    dataset=[]
    targets=[]
    env_indices = []
    for m in range(0, path_length-1):
        data = np.concatenate( (path[m], path[path_length-1]) ).astype(np.float32)
        targets.append(path[m+1])
        dataset.append(data)
        env_indices.append(obs_i)
    return dataset,targets,env_indices

def neural_replan(path, obc, obs, step_sz=DEFAULT_STEP):
    # replan segments of paths
    new_path = []
    new_path.append(path[0])
    # rule out nodes that are already in collision
    for i in range(1,len(path)-1):
        if not IsInCollision(path[i],obc):
            new_path.append(path[i])
    new_path.append(path[-1])
    path = new_path
    new_path = [path[0]]
    for i in range(len(path)-1):
        # look at if adjacent nodes can be connected
        # assume start is already in new path
        start = path[i]
        goal = path[i+1]
        steer = steerTo(start, goal, obc, step_sz=step_sz)
        if steer:
            new_path.append(goal)
        else:
            # plan mini path
            mini_path = neural_replanner(start, goal, obc, obs, step_sz=step_sz)
            if mini_path:
                new_path += mini_path[1:]  # take out start point
            else:
                new_path += path[i+1:]     # just take in the rest of the path
                break
    return new_path


def neural_replanner(start, goal, obc, obs, step_sz=DEFAULT_STEP):
    # plan a mini path from start to goal
    MAX_LENGTH = 50
    itr=0
    pA=[]
    pA.append(start)
    pB=[]
    pB.append(goal)
    target_reached=0
    tree=0
    new_path = []
    while target_reached==0 and itr<MAX_LENGTH:
        itr=itr+1  # prevent the path from being too long
        if tree==0:
            ip1=torch.cat((obs,start,goal)).unsqueeze(0)
            ip1=to_var(ip1)
            start=mpNet(ip1).squeeze(0)
            start=start.data.cpu()
            pA.append(start)
            tree=1
        else:
            ip2=torch.cat((obs,goal,start)).unsqueeze(0)
            ip2=to_var(ip2)
            goal=mpNet(ip2).squeeze(0)
            goal=goal.data.cpu()
            pB.append(goal)
            tree=0
        target_reached=steerTo(start, goal, obc, step_sz=step_sz)
    if target_reached==0:
        return 0
    else:
        for p1 in range(len(pA)):
            new_path.append(pA[p1])
        for p2 in range(len(pB)-1,-1,-1):
            new_path.append(pB[p2])
    return new_path

def complete_replan(path, true_path, true_path_length, obc, obs, obs_i, data_all, step_sz=DEFAULT_STEP):
    global num_trained_samples
    # complete replan part of the paths, and use the segments for training
    # replan segments of paths
    # input path: list of tensor
    # obs: tensor
    new_path = []
    new_path.append(path[0])
    # rule out nodes that are already in collision
    for i in range(1, len(path)-1):
        if not IsInCollision(path[i],obc):
            new_path.append(path[i])
    new_path.append(path[-1])
    path = new_path
    new_path = [path[0]]
    print('path length: %d' % (len(path)))
    for i in range(len(path)-1):
        # look at if adjacent nodes can be connected
        # assume start is already in new path
        start = path[i]
        goal = path[i+1]
        steer = steerTo(start, goal, obc, step_sz=step_sz)
        if steer:
            new_path.append(goal)
            print('bingo, steerTo succeed!')
        else:
            # firstly check if only start and goal exists
            # if so, then just use training data
            # true_path is numpy array
            if len(path) == 2:
                mini_path = true_path[:true_path_length]
                mini_path = [torch.from_numpy(p).type(torch.FloatTensor) for p in mini_path]
            else:
                # plan mini path
                mini_path = complete_replanner(start, goal, obc, obs, step_sz=step_sz)
                mini_path = [torch.from_numpy(p).type(torch.FloatTensor) for p in mini_path]
            new_path += mini_path[1:]  # take out start point
            # transform mini_path to training data, and train the network
            # also put the training data into replay memory
            mini_path = [p.numpy() for p in mini_path]
            dataset, targets, env_indices = transformToTrain(mini_path, len(mini_path), obs, obs_i)
            print('inside rrt*, targets size: %d' % (len(targets)))
            num_trained_samples += len(targets)
            data_all += list(zip(dataset,targets,env_indices))
            bi = np.concatenate( (obs.numpy().reshape(1,-1).repeat(len(dataset),axis=0), dataset), axis=1).astype(np.float32)
            bi = torch.FloatTensor(bi)
            bt = torch.FloatTensor(targets)
            mpNet.zero_grad()
            bi=to_var(bi)
            bt=to_var(bt)
            mpNet.observe(bi, 0, bt)
    return new_path

def complete_replanner(start, goal, obc, obs, step_sz=DEFAULT_STEP):
    obstacleList = [
    (obc[0][0], obc[0][1], 5.0, 5.0),
    (obc[1][0], obc[1][1], 5.0, 5.0),
    (obc[2][0], obc[2][1], 5.0, 5.0),
    (obc[3][0], obc[3][1], 5.0, 5.0),
    (obc[4][0], obc[4][1], 5.0, 5.0),
    (obc[5][0], obc[5][1], 5.0, 5.0),
    (obc[6][0], obc[6][1], 5.0, 5.0),
    ]
    p_cost=0.0
    # for simple2d randarea: -20~20
    # for 3d: -40~40
    rrt = RRT(start=start.numpy(), goal=goal.numpy(),randArea=[-20, 20], obstacleList=obstacleList, p_cost=p_cost)
    path, cost = rrt.Planning()
    if path ==0:
        print('path not found')
    path = path[1:]
    path = [np.array(p) for p in path]
    return path

def main(args):
    # Create model directory
    global mpNet, num_trained_samples
    md_type = 'deep'
    if not args.AEtype_deep:
        md_type = 'simple'
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    mpNet = End2EndMPNet(args.mlp_input_size, args.output_size, md_type, \
                args.n_tasks, args.n_memories, args.memory_strength, args.grad_step)
    if args.maml:
        # load encoder model from maml_path
        mpNet.encoder.load_state_dict(torch.load(args.maml_path))
    model_path='mpNet_cont_normal_replay_epoch_%d.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
        mpNet.set_opt(torch.optim.Adagrad, lr=1e-2)
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))

    # load train and test data
    print('loading...')

    obc,obs,paths,path_lengths = load_raw_dataset(N=args.no_env, NP=args.no_motion_paths, folder=args.data_path)
    obs = torch.from_numpy(obs).type(torch.FloatTensor)
    # Pretrain the Models
    print('continual training...')
    for epoch in range(1,args.num_epochs+1):
        data_all = []
        num_path_trained = 0
        time_env = []
        num_path_pretrained = 0
        print('epoch' + str(epoch))
        for i in range(len(paths)):
            time_path = []
            for j in range(len(paths[i])):
                time0 = time.time()
                print('epoch: %d, training... env: %d, path: %d' % (epoch, i+1, j+1))
                if path_lengths[i][j] == 0:
                    continue
                fp = False
                path = [torch.from_numpy(paths[i][j][0]).type(torch.FloatTensor),\
                        torch.from_numpy(paths[i][j][path_lengths[i][j]-1]).type(torch.FloatTensor)]
                step_sz = DEFAULT_STEP
                # pretrain
                if num_path_pretrained < args.pretrain_path:
                    # pretrain
                    print('epoch: %d, pretraining... env: %d, path: %d' % (epoch, i+1, j+1))
                    pretrain_path = paths[i][j][:path_lengths[i][j]]  # numpy
                    dataset, targets, env_indices = transformToTrain(pretrain_path, \
                                                    len(pretrain_path), obs[i], i)
                    num_trained_samples += len(targets)
                    data_all += list(zip(dataset,targets,env_indices))
                    bi = np.concatenate( (obs[i].numpy().reshape(1,-1).repeat(len(dataset),axis=0), dataset), axis=1).astype(np.float32)
                    bi = torch.FloatTensor(bi)
                    bt = torch.FloatTensor(targets)
                    mpNet.zero_grad()
                    bi=to_var(bi)
                    bt=to_var(bt)
                    mpNet.observe(bi, 0, bt)
                    num_path_pretrained += 1
                    continue

                # hybrid train
                # Note that path are list of tensors
                for t in range(args.MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                    if (t == 2):
                        step_sz = 0.04
                    elif (t == 3):
                        step_sz = 0.03
                    elif (t > 3):
                        step_sz = 0.02

                    path = neural_replan(path, obc[i], obs[i], step_sz=step_sz)
                    path = lvc(path, obc[i],step_sz=step_sz)
                    if feasibility_check(path,obc[i],step_sz=0.01):
                        fp = True
                        print('feasible, ok!')
                        break
                print('number of samples trained up to now: %d' % (num_trained_samples))
                if not fp:
                    print('using RRT*...')
                    # since this is complete replan, we are using the finest step size
                    path = complete_replan(path, paths[i][j], path_lengths[i][j], \
                                           obc[i], obs[i], i, data_all, step_sz=0.01)
                    #print(data_all)
                    num_path_trained += 1
                    # perform rehersal when certain number of batches have passed
                    if num_path_trained % args.freq_rehersal == 0 and len(data_all) > args.batch_rehersal:
                        print('rehersal...')
                        sample = random.sample(data_all, args.batch_rehersal)
                        dataset, targets, env_indices = list(zip(*sample))
                        dataset, targets, env_indices = list(dataset), list(targets), list(env_indices)
                        bi = np.concatenate( (obs[env_indices], dataset), axis=1).astype(np.float32)
                        bt = targets
                        bi = torch.FloatTensor(bi)
                        bt = torch.FloatTensor(bt)
                        mpNet.zero_grad()
                        bi=to_var(bi)
                        bt=to_var(bt)
                        mpNet.observe(bi, 0, bt, False)  # train but don't remember
                time_spent = time.time() - time0
                time_path.append(time_spent)
                print('it takes time: %f s' % (time_spent))
            time_env.append(time_path)
        print('number of samples trained in total: %d' % (num_trained_samples))

        # Save the models
        if epoch > 0:
            model_path='mpNet_cont_train_epoch_%d.pkl' %(epoch)
            save_state(mpNet, os.path.join(args.model_path,model_path))
            num_train_sample_record = args.model_path+'num_trained_samples_epoch_%d.txt' % (epoch)
            num_train_path_record = args.model_path+'num_trained_paths_epoch_%d.txt' % (epoch)
            f = open(num_train_sample_record, 'w')
            f.write('%d\n' % (num_trained_samples))
            f.close()
            f = open(num_train_path_record, 'w')
            f.write('%d\n' % (num_path_trained))
            f.close()
            pickle.dump(time_env, open(args.model_path+'planning_time_epoch_%d.txt' % (epoch), "wb" ))
            # test

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

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--seen', type=int, default=0, help='seen or unseen? 0 for seen, 1 for unseen')
parser.add_argument('--AEtype_deep', type=int, default=1, help='indicate that autoencoder is deep model')
parser.add_argument('--maml', type=int, default=1, help='load parameters trained with maml')
parser.add_argument('--maml_path', type=str, default='../results/', help='path for loading encoder model')
parser.add_argument('--device', type=int, default=0, help='cuda device')

parser.add_argument('--batch_path', type=int,default=10,help='number of optimal paths in each environment')
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--freq_rehersal', type=int, default=20, help='after how many paths perform rehersal')
parser.add_argument('--batch_rehersal', type=int, default=100, help='rehersal on how many data (not path)')
parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--MAX_NEURAL_REPLAN', type=int, default=1)

parser.add_argument('--pretrain_path', type=int, default=200, help='number of paths for pretraining before hybrid train')
args = parser.parse_args()
print(args)
main(args)
