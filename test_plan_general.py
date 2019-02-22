from plan_general import *
from Model.GEM_end2end_model import End2EndMPNet
from utility import *
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
world_size=20
# test neural replan
mpnet = End2EndMPNet(mlp_input_size=28+4, output_size=2, AEtype='deep', \
             n_tasks=1, n_memories=10, memory_strength=0.5, grad_step=1)
path = [torch.tensor([0.,6.]).float(), torch.tensor([0.,-6.]).float()]
print(path)
DEFAULT_STEP = 0.05
step_sz = DEFAULT_STEP
for t in range(11):
    if (t == 2):
        step_sz = 0.04
    elif (t == 3):
        step_sz = 0.03
    elif (t > 3):
        step_sz = 0.02
    print('%d-th planning...' % (t))
    unnormalize_func = lambda x: unnormalize(x, world_size)
    path = neural_replan(mpnet, path, obc, obs, IsInCollision, unnormalize_func, step_sz=DEFAULT_STEP)
    path = lvc(path, obc, IsInCollision, step_sz=step_sz)
    print(path)
    if feasibility_check(path,obc, IsInCollision, step_sz=0.01):
        fp = True
        print('feasible, ok!')
        break
print(path)
