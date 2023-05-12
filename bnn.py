from torchbnn.modules.linear import BayesLinear
from torchbnn.modules.module import BayesModule
from torchbnn.utils import freeze, unfreeze
from torch.nn.functional import relu


'''
this class should just implement the base bnn and backprop step for it
consider the dynamics module for a complete dynamics model (that contains a set of bnn) 
'''

class BNN(BayesModule):
    def __init__(self, action_dim, obs_dim, reward_dim, W_world_model=None):
        super(BayesModule, self).__init__()

        # is reward dim == 1 ???
        self.in_features = action_dim + obs_dim
        self.h_in_features = self.in_features + 64
        self.h_out_features = self.h_in_features + 32
        self.out_features = obs_dim + reward_dim

        self.input_layer = BayesLinear(prior_mu=0, prior_sigma=1,
                                       in_features=self.in_features,
                                       out_features=self.h_in_features)
        self.hidden_layer = BayesLinear(prior_mu=0, prior_sigma=1,
                                        in_features=self.h_in_features,
                                        out_features=self.h_out_features)
        self.ouput_layer = BayesLinear(prior_mu=0, prior_sigma=1,
                                       in_features=self.h_out_features,
                                       out_features=self.out_features)

        if W_world_model:
            self.copy_params_from_world_model(W_world_model)

    def forward(self, x):
        x = relu(self.input_layer(x))
        x = relu(self.hidden_layer(x))
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



