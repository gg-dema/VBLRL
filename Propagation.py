import torch
import torch.nn as nn
import torch.nn.functional as F



class LinNet(nn.Module):

    def __init__(self, action_dim=4, obs_dim=39, reward_dim=1):
        super(LinNet, self).__init__()

        self.in_features = action_dim + obs_dim
        self.h1_in_features = 128
        self.h1_out_features = 256
        self.h2_in_features = self.h1_out_features
        self.h2_out_features = 128
        self.out_features = obs_dim + reward_dim

        self.input_layer = nn.Linear(self.in_features, self.h1_in_features)
        self.hidden1_layer = nn.Linear(self.h1_in_features, self.h1_out_features)
        self.hidden2_layer = nn.Linear(self.h2_in_features, self.h2_out_features)
        self.output_layer = nn.Linear(self.h2_out_features, self.out_features)

        for module in self.parameters():
            module.requires_grad = False

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden1_layer(x))
        x = F.relu(self.hidden2_layer(x))
        return self.output_layer(x)


class Propagation_net:
    def __init__(self, num_particles=50, action_dim=4, obs_dim=39):

        self.num_particles = num_particles
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.deterministic_nets = []
        for i in range(self.num_particles):
            self.deterministic_nets.append(LinNet())

    def move_to_gpu(self):
        for net_idx in range(self.num_particles):
            self.deterministic_nets[net_idx] = self.deterministic_nets[net_idx].to('cuda:0')

    def sample_from(self, bnn):
        for net in self.deterministic_nets:
            iterator_layer_bnn = bnn.named_modules()
            _ = next(iterator_layer_bnn)  # first reference : entire class, discard
            for name, layer in iterator_layer_bnn:
                weight, bias = layer.sample_weight(requires_grad=False)
                net.get_submodule(name).weight = nn.Parameter(weight)
                net.get_submodule(name).bias = nn.Parameter(bias)


    def propagate(self, initial_state, actions, dev='cuda:0'):
        """
        :param initial_state: the original state, is common to all future net
        :param actions: this comes from the cem: [ 1 x (act*horizons)]  <--- this will be call for pop size
        :return: Q_val
        """
        X = torch.zeros((self.num_particles, self.obs_dim + self.action_dim), device=dev)
        Y = torch.zeros((self.num_particles, self.obs_dim + 1), device=dev)  # 1 AKA reward
        X[:, :self.obs_dim] = initial_state
        rewards = torch.zeros(self.num_particles, device=dev)

        with torch.no_grad():

            for h in range(actions.shape[0] // 4):  # AKA horizon
                # add action to propagate
                X[:, self.obs_dim:] = actions[h * self.action_dim: (h + 1) * self.action_dim]
                for row_idx, x in enumerate(X):
                    Y[row_idx] = self.deterministic_nets[row_idx](x)
                X[:, :self.obs_dim] = Y[:, :self.obs_dim]  # update state
                rewards += Y[:, -1]  # collect rewards

        return (rewards / h).mean()



if __name__ == "__main__":
    import time
    from bnn import BNN
    from cem_optimizer_v2 import CEM_opt
    import numpy as np

    torch.set_default_dtype(torch.float64)

    dev = 'cuda:0'
    num_particles = 50
    cem = CEM_opt(num_particles)
    t = time.time()
    act_sequences = torch.from_numpy(cem.population)
    r = np.zeros(act_sequences.shape[0])
    print(time.time()-t)
    original_model = BNN(action_dim=4, obs_dim=39, reward_dim=1)
    bnn_path = '/home/dema/PycharmProjects/lifelong_rl/VBLRL_rl_exam/model_stock/world/model_envWorld.pth'
    original_model.load_state_dict(torch.load(bnn_path, map_location=torch.device('cpu')))
    prop_net = Propagation_net(num_particles)
    init_s = torch.randn(39)


    '''
    # incredibilmente lento, l'ho fatto andare per un 4 minuti, non so a che punto era,
    # non va ne con le deepcopy ne senza
    
    from multiprocessing import Process
    from copy import deepcopy
    
    def funct_to_parallelize(prop_net, bnn, init_s, act_seq, rew, idx):
        prop_net.sample_from(bnn)
        r = prop_net.propagate(init_s, act_seq, dev='cpu')
        rew[idx] = r.detach().numpy()

    process = []
    t = time.time()
    for i, act_seq in enumerate(act_sequences):
        p = Process(target=funct_to_parallelize, args=(deepcopy(prop_net), deepcopy(original_model), init_s, act_seq, r, i))
        p.start()
        process.append(p)

    for p in process:
        p.join()
    
    print("multithread cpu: ", time.time() - t)
    print(r != 0)
    '''



    t = time.time()
    for idx, act_seq in enumerate(act_sequences):
        prop_net.sample_from(original_model)
        r[idx] = prop_net.propagate(init_s, act_seq, dev='cpu')
        # here update the cem
    print('cpu: ', time.time() - t)
    print(r)



    '''
    init_s = init_s.to(dev)
    original_model = original_model.to(dev)
    prop_net.model_to_gpu()
    t = time.time()
    for act_seq in act_sequences:
        act_seq = act_seq.to(dev)
        #prop_net.sample_from(original_model)
        prop_net.propagate(init_s, act_seq)
    print('gpu: ', time.time() - t)
    '''


    # time to add :
    #
    #
    # 50 particles by colab
    # cpu:  0.13980364799499512
    # gpu:  0.21935749053955078
    #
    # 50 particles by my pc:
    # cpu: 0.039
    # gpu: 0.38
    #
    # 50 particles old method (fake threads):
    # cpu: 0.1349
    # gpu: idk
    #
    # FULL POWER (all correct parameters : 500 particles)
    # total rollout in cpu : 34 secondi
    # total rollout in gpu : 37 secondi
    # total rollout in cpu fake threads: 72 secondi
