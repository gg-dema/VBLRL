import torch
import torch.nn as nn
import torch.nn.functional as F

class LinNet(nn.Module):

    def __init__(self):
        super(LinNet, self).__init__()

        self.l1 = nn.Linear(43, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 40)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return F.tanh(self.l4(x))


class Propagation_net:
    def __init__(self, num_particles=50, action_dim=4, obs_dim=39):

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

    def propagate(self, initial_state, actions, dev='cuda:0'):
        """
        :param initial_state: the original state, is common to all future net
        :param actions: this comes from the cem: [ 1 x (act*horizons)]  <--- this will be call for pop size
        :return: mean Q_val for net aka mean of rewards
        """
        X = torch.zeros((self.num_particles, self.obs_dim + self.action_dim), device=dev)
        Y = torch.zeros((self.num_particles, self.obs_dim + 1), device=dev)  # 1 AKA reward
        X[:, :self.obs_dim] = initial_state
        rewards = torch.zeros(self.num_particles, device=dev)

        with torch.no_grad():

            for h in range(actions.shape[0]//4):  # AKA horizon
                # add action to propagate
                a = actions[h*self.action_dim: h+1*self.action_dim]
                X[:, self.obs_dim:] = actions[h*self.action_dim: (h+1)*self.action_dim]
                for row_idx, x in enumerate(X):
                    Y[row_idx] = self.deterministic_nets[row_idx](x)

                X[:, :self.obs_dim] = Y[:, :self.obs_dim]   # update state
                rewards += Y[:, -1]                         # collect rewards

        return (rewards/h).mean()


    def infer(self, init_state):
        pass
        '''
        reward = torch.tensor(INSERIRE DIM)
        for action_seq_idx, seq in enumerate(...) :  #call the cem? pass by params? btw, iterate over population
            self.sample_nets() # load weight for all the net
            reward[action_seq_idx] = self.propagate(init_state, seq)
        # update cem

        # do i want to do a separeate plan step where i take the elite action?
        # should i do all here?
        '''
    def sample_weight(self):
      w = []
      for model in self.deterministic_nets:
        w.append(model.state_dict())
      return w
    
    def load_sample(self, w):
      for id, parameters in enumerate(w):
        self.deterministic_nets[id].load_state_dict(parameters)

if __name__ == "__main__":

  import time
  dev = 'cuda:0'


  prop_net = Propagation_net()
  W = prop_net.sample_weight()

  init_s = torch.randn(39)
  act_seq = torch.randn(4*20)

  t = time.time()
  prop_net.load_sample(W)
  prop_net.propagate(init_s, act_seq, dev='cpu')
  print('cpu: ', time.time()-t )



  init_s = init_s.to(dev)
  act_seq = act_seq.to(dev)
  prop_net.model_to_gpu()

  t = time.time()
  prop_net.load_sample(W)
  prop_net.propagate(init_s, act_seq)
  print('gpu: ', time.time() - t )

  # time to add :
  # for each plan step: pop size = 500 ---> 500 x propagation time
  # + load all 50 model for each step of 500
  # 
  # cpu:  0.13980364799499512
  # gpu:  0.21935749053955078
