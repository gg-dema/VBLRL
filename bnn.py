from collections import OrderedDict

from torchbnn.modules.linear import BayesLinear
from torchbnn.modules.module import BayesModule
from torchbnn.utils import freeze, unfreeze
from torch.nn.functional import relu
import torch.nn.functional as F
import torch

class BayesLayerWithSample(BayesLinear):
    """
        same of BayesLinear from torchbnn, just add the possibility of sample
        a set of weight from the net. Functional paradigm for calc linear output
    """
    def sample_layer_functional(self, device):
        if self.weight_eps is None:
            weight = (self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(
                self.weight_log_sigma)).detach()
        else:
            weight = (self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps).detach()
        weight = weight.to(device)

        if self.bias:
            if self.bias_eps is None:
                bias = (self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)).detach()
            else:
                bias = (self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps).detach()
            bias = bias.to(device)
        else:
            bias = None

        def linear_step(x):
            return F.linear(x, weight, bias)

        return linear_step

    def sample_weight(self, requires_grad=True):
        # remove device as args
        weight, bias = None, None

        if self.weight_eps is None:
            weight = (self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(
                self.weight_log_sigma)).detach()
        else:
            weight = (self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps).detach()
        weight.requires_grad = requires_grad

        if self.bias:
            if self.bias_eps is None:
                bias = (self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)).detach()
            else:
                bias = (self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps).detach()
            bias.requires_grad = requires_grad

        return weight, bias



class BNN(BayesModule):
    def __init__(self, action_dim, obs_dim, reward_dim, weight=None):
        super(BayesModule, self).__init__()

        self.in_features = action_dim + obs_dim
        self.out_features = obs_dim + reward_dim

        self.input_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1, in_features=self.in_features, out_features=128)
        self.hidden1_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1, in_features=128, out_features=256)
        self.hidden2_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1, in_features=256, out_features=256)
        self.hidden3_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1, in_features=256, out_features=512)
        self.hidden4_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1, in_features=512, out_features=512)
        self.output_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1,  in_features=512, out_features=self.out_features)

        if weight:
            self.copy_params_from_model(weight)

    def forward(self, x):
        x = relu(self.input_layer(x))
        x = relu(self.hidden1_layer(x))
        x = relu(self.hidden2_layer(x))
        x = relu(self.hidden3_layer(x))
        x = relu(self.hidden4_layer(x))
        x = self.output_layer(x)
        return x

    def copy_params_from_model(self, W):
        try:
            self.load_state_dict(W)
        except BaseException:
            print('non compatible W')

    def sample_linear_net_functional(self, device):
        step = []
        for layer in self._modules.items():
            step.append(layer[1].sample_layer_functional(device))

        def forward_with_sample(x):
            for op in step:
                x = F.relu(op(x))
            return x

        return forward_with_sample

    # forse rimuovere questa e' una buona idea
    def sample_linear_net_weight(self):
        params = OrderedDict()
        for name, layer in self._modules.items():
            dict_forlayer = layer.sample_weight()
            for tensor_name, tensor in dict_forlayer.items():
                params[name+'.'+tensor_name] = tensor
        return params

    def deterministic_mode(self):
        '''deterministic output'''
        freeze(self)

    def stochatisc_mode(self):
        '''stochatisc output'''
        unfreeze(self)



if __name__ == '__main__':
    bnn = BNN(4, 39, 1)
