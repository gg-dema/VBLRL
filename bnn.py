from torchbnn.modules.linear import BayesLinear
from torchbnn.modules.module import BayesModule
from torchbnn.utils import freeze, unfreeze
from torch.nn.functional import relu


'''
this class should just implement the base bnn and backprop step for it
consider the dynamics module for a complete dynamics model (that contains a set of bnn) 
'''

class BNN(BayesModule):
    def __init__(self, action_dim, obs_dim, reward_dim, weight_world_model=None):
        super(BayesModule, self).__init__()

        self.in_features = action_dim + obs_dim
        self.h1_in_features = 128
        self.h1_out_features = 256
        self.h2_in_features = self.h1_out_features
        self.h2_out_features = 128
        self.out_features = obs_dim + reward_dim

        self.input_layer = BayesLinear(prior_mu=0, prior_sigma=1,
                                       in_features=self.in_features,
                                       out_features=self.h1_in_features)
        self.hidden1_layer = BayesLinear(prior_mu=0, prior_sigma=1,
                                         in_features=self.h1_in_features,
                                         out_features=self.h1_out_features)

        self.hidden2_layer = BayesLinear(prior_mu=0, prior_sigma=1,
                                         in_features=self.h2_in_features,
                                         out_features=self.h2_out_features)
        self.ouput_layer = BayesLinear(prior_mu=0, prior_sigma=1,
                                       in_features=self.h2_out_features,
                                       out_features=self.out_features)

        if weight_world_model:
            self.copy_params_from_world_model(weight_world_model)

    def forward(self, x):
        x = relu(self.input_layer(x))
        x = relu(self.hidden1_layer(x))
        x = relu(self.hidden2_layer(x))
        x = self.ouput_layer(x)
        return x

    def copy_params_from_world_model(self, W):
        try:
            self.load_state_dict(W)
        except BaseException:
            print('non compatible W')

    def deterministic_mode(self):
        '''deterministic output'''
        freeze(self)

    def stochatisc_mode(self):
        '''stochatisc output'''
        unfreeze(self)


if __name__ == '__main__':
    import torch

    basic_bnn = BNN(
        action_dim=10,
        obs_dim=10,
        reward_dim=10
    )

    x = torch.rand((20, ))
    # here the 2 output should be stochastic
    out1 = basic_bnn(x)
    out2 = basic_bnn(x)
    print(f'{out1=}')
    print(f'{out2=}')

    basic_bnn.deterministic_mode()

    out1 = basic_bnn(x)
    out2 = basic_bnn(x)
    print(f'{out1=}')
    print(f'{out2=}')
