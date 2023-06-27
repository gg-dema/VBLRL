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
    def __init__(self, action_dim, obs_dim, reward_dim, weight_world_model=None):
        super(BayesModule, self).__init__()

        self.in_features = action_dim + obs_dim
        self.h1_in_features = 128
        self.h1_out_features = 256
        self.h2_in_features = self.h1_out_features
        self.h2_out_features = 128
        self.out_features = obs_dim + reward_dim

        self.input_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1,
                                          in_features=self.in_features,
                                          out_features=self.h1_in_features)
        self.hidden1_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1,
                                            in_features=self.h1_in_features,
                                            out_features=self.h1_out_features)

        self.hidden2_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1,
                                            in_features=self.h2_in_features,
                                            out_features=self.h2_out_features)
        self.output_layer = BayesLayerWithSample(prior_mu=0, prior_sigma=1,
                                          in_features=self.h2_out_features,
                                          out_features=self.out_features)

        if weight_world_model:
            self.copy_params_from_world_model(weight_world_model)

    def forward(self, x):
        x = relu(self.input_layer(x))
        x = relu(self.hidden1_layer(x))
        x = relu(self.hidden2_layer(x))
        x = self.output_layer(x)
        return x

    def copy_params_from_world_model(self, W):
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
    import time
    dev = 'cuda:0'
    test_freeze_unfreeze = False
    test_functionalSample_vs_basicNet = False

    basic_bnn_gpu = BNN(
        action_dim=10,
        obs_dim=10,
        reward_dim=10
    ).to(dev)

    basic_bnn_cpu = BNN(
        action_dim=10,
        obs_dim=10,
        reward_dim=10
    )

    x_gpu = torch.rand((20,)).to(dev)
    x_cpu = torch.rand((20, ))

    if test_freeze_unfreeze:
        # here the 2 output should be stochastic
        basic_bnn_gpu.stochatisc_mode()
        out1 = basic_bnn_gpu(x_gpu)
        out2 = basic_bnn_gpu(x_gpu)
        print(f'{out1=}, dev = {out1.get_device()}')
        print(f'{out2=}, dev = {out2.get_device()}')

        basic_bnn_gpu.deterministic_mode()

        out1 = basic_bnn_gpu(x_gpu)
        out2 = basic_bnn_gpu(x_gpu)
        print(f'{out1=}')
        print(f'{out2=}')

        # restore old behaviour
        basic_bnn_gpu.stochatisc_mode()

    if test_functionalSample_vs_basicNet:
        basic_bnn_gpu.deterministic_mode()
        basic_bnn_cpu.deterministic_mode()

        iter_numb = 1000
        t_start = time.time()
        for i in range(iter_numb):
            _ = basic_bnn_cpu(x_cpu)
        t_end = time.time()
        has_grad = _.requires_grad
        print(f'{iter_numb} step forward cpu NO SAMPLE:{t_end - t_start}, grad: {has_grad}')

        t_start = time.time()
        for i in range(iter_numb):
            _ = basic_bnn_gpu(x_gpu)
        t_end = time.time()
        has_grad = _.requires_grad
        print(f'{iter_numb} step forward gpu NO SAMPLE:{t_end - t_start}, grad: {has_grad}')



        sample_cpu = basic_bnn_cpu.sample_linear_net_functional(device='cpu')
        t_start = time.time()
        for i in range(iter_numb):
            _ = sample_cpu(x_cpu)
        t_end = time.time()
        has_grad = _.requires_grad
        print(f'{iter_numb} step forward cpu FUNCT:{t_end - t_start}, grad: {has_grad}')



        sample_gpu = basic_bnn_gpu.sample_linear_net_functional(device='cuda:0')
        t_start = time.time()
        for i in range(iter_numb):
            _ = sample_gpu(x_gpu)
        t_end = time.time()
        has_grad = _.requires_grad
        print(f'{iter_numb} step forward gpu FUNCT:{t_end - t_start}, grad: {has_grad}')

    print(basic_bnn_cpu.sample_linear_net_weight('cpu'))
