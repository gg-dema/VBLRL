import torch
import torch.nn as nn
import torch.nn.functional as F

class LinNet(nn.Module):

    def __init__(self):
        super(LinNet, self).__init__()

        self.l1 = nn.Linear(10, 50)
        self.l2 = nn.Linear(50, 100)
        self.l3 = nn.Linear(100, 400)
        self.l4 = nn.Linear(400, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return F.tanh(self.l4(x))


class Propagation_net:
    def __init__(self, num_particles=40, action_dim=4, obs_dim=39):

        self.num_particles = num_particles
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.deterministic_nets = nn.ModuleList()
        for i in range(self.num_particles):
            self.deterministic_nets.append(LinNet())

    def model_to_gpu(self):
        if torch.cuda.is_available():
            dev = 'cuda:0'
            for net_idx in range(self.num_particles):
                self.deterministic_nets[net_idx] = self.deterministic_nets[net_idx].to(dev)
        else:
            print('no gpu here')

    def propagate(self, initial_state, actions):
        """
        :param initial_state: the original state, is common to all future net
        :param actions: this comes from the cem: [ 1 x (act*horizons)]  <--- this will be call for pop size
        :return: mean Q_val for net aka mean of rewards
        """
        X = torch.zeros((self.num_particles, self.obs_dim + self.action_dim))
        Y = torch.zeros((self.num_particles, self.obs_dim + 1))  # 1 AKA reward
        X[:, :self.obs_dim] = initial_state
        rewards = torch.zeros(self.num_particles)

        with torch.no_grad():

            for h in range(actions.shape[1]//4):  # AKA horizon
                # add action to propagate
                X[:, self.obs_dim:] = actions[h*self.action_dim: h+1*self.action_dim]
                for row_idx, x in enumerate(X):
                    Y[row_idx] = self.deterministic_nets[row_idx](x)

                X[:, :self.obs_dim] = Y[:, :self.obs_dim]   # update state
                rewards += Y[:, -1]                         # collect rewards

        return (rewards/h).mean()


    def infer(self, init_state):

        reward = torch.tensor(INSERIRE DIM)
        for action_seq_idx, seq in enumerate(...) :  #call the cem? pass by params? btw, iterate over population
            self.sample_nets() # load weight for all the net
            reward[action_seq_idx] = self.propagate(init_state, seq)
        # update cem

        # do i want to do a separeate plan step where i take the elite action?
        # should i do all here?








if __name__ == '__main__':
    import time
    dev = 'cuda:0'
    lin_net1 = LinNet()
    lin_net2 = LinNet()
    lin_net1 = lin_net1.to(dev)
    lin_net2 = lin_net2.to(dev)
    x = torch.randn(10).to(dev)
    t = time.time()
    lin_net1(x)
    lin_net2(x)
    print('2 separate model : ', time.time() - t)
    #print(out_base.shape, out_base)
    time.sleep(1)

    x1, x2 = torch.randn(10), torch.randn(10)
    multinet = Propagation_net()
    multinet.model_to_gpu()
    #x1, x2 = x1.to(dev), x2.to(dev)
    t = time.time()
    #multinet.fake_forward(x1, x2)
    print('only 1 model :', time.time() - t)
    time.sleep(1)
