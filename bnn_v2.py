import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _elements(t: torch.Tensor):
    return np.prod(t.shape)


class _BayesianLinerLayer(nn.Module):
    """A linear layer which samples network parameters on forward calculation.
    Local re-parameterization trick is used instead of direct sampling of network parameters.
    """

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        deterministic=False,
        weight_out=0.1,
        device="cpu",
    ):
        super().__init__()
        self._fan_in, self._fan_out = fan_in, fan_out
        self.deterministic = deterministic

        self._W_mu = torch.normal(
            torch.zeros(fan_in, fan_out), torch.ones(fan_in, fan_out)
        ).to(device)
        self._b_mu = torch.normal(torch.zeros(fan_out), torch.ones(fan_out)).to(device)

        self._b_rho = torch.log(np.exp(torch.ones(fan_out) * 0.5) - 1.0).to(device)
        self._W_rho = torch.log(torch.exp(torch.ones(fan_in, fan_out) * 0.5) - 1.0).to(
            device
        )

        self.weight_out = weight_out

        self.W_mu_old = torch.Tensor(fan_in, fan_out).detach()
        self.W_rho_old = torch.Tensor(fan_in, fan_out).detach()

        self.b_mu_old = torch.Tensor(
            fan_out,
        ).detach()
        self.b_rho_old = torch.Tensor(
            fan_out,
        ).detach()

        self._W_var, self._b_var = self._rho2var(self._W_rho), self._rho2var(
            self._b_rho
        )

        self._parameter_number = _elements(self._W_mu) + _elements(self._b_mu)
        self._distributional_parameter_number = (
            self._parameter_number + _elements(self._b_mu) + _elements(self._b_rho)
        )
        self.save_old_params()

    @staticmethod
    def _rho2var(rho):
        return torch.log(1.0 + torch.exp(rho)).pow(2)

    @property
    def parameter_number(self):
        return self._parameter_number

    @property
    def distributional_parameter_number(self):
        return self._distributional_parameter_number

    def get_parameters(self):
        """Return all parameters in this layer as vectors of mu and rho."""
        params_mu = torch.cat(
            [self._W_mu.data.reshape(-1), self._b_mu.data.reshape(-1)]
        )
        params_rho = torch.cat(
            [self._W_rho.data.reshape(-1), self._b_rho.data.reshape(-1)]
        )
        return params_mu, params_rho

    def get_parameters_old(self):
        """Return all parameters in this layer as vectors of mu and rho."""
        params_mu = torch.cat(
            [self.W_mu_old.data.reshape(-1), self.b_mu_old.data.reshape(-1)]
        )
        params_rho = torch.cat(
            [self.W_rho_old.data.reshape(-1), self.b_rho_old.data.reshape(-1)]
        )
        return params_mu, params_rho

    def set_parameters(self, params_mu: torch.Tensor, params_rho: torch.Tensor):
        """Receive parameters (mu and rho) as vectors and set them."""

        self._W_mu = params_mu[: _elements(self._W_mu)].reshape(self._W_mu.size())
        self._b_mu = params_mu[_elements(self._W_mu) :].reshape(self._b_mu.size())

        self._W_rho = params_rho[: _elements(self._W_rho)].reshape(self._W_rho.size())
        self._b_rho = params_rho[_elements(self._W_rho) :].reshape(self._b_rho.size())

        self._W_var, self._b_var = self._rho2var(self._W_rho), self._rho2var(
            self._b_rho
        )

    def forward(self, X, share_paremeters_among_samples=True):
        """
        Linear forward calculation with local re-parameterization trick.
        params
        ---
        X: (batch, input_size)
        share_paremeters_among_samples: (bool) Use the same set of parameters for samples in a batch

        return
        ---
        r: (batch, output_size)
        """
        gamma = X @ self._W_mu + self._b_mu
        delta = X.pow(2) @ self._W_var + self._b_var

        if self.deterministic:
            r = gamma
        else:
            if share_paremeters_among_samples:
                zeta = torch.randn(1, self._fan_out).expand(X.size(0), self._fan_out)
            else:
                zeta = torch.randn_like(gamma)
            zeta = zeta.to(X.device)
            r = gamma + delta.pow(0.5) * zeta * self.weight_out
        return r

    def save_old_params(self):
        self.W_mu_old = self._W_mu.clone()
        self.W_rho_old = self._W_rho.clone()
        self.b_mu_old = self._b_mu.clone()
        self.b_rho_old = self._b_rho.clone()


class BNN:
    def __init__(
        self,
        observation_size,
        action_size,
        reward_size,
        hidden_layers,
        hidden_layer_size,
        max_logvar,
        min_logvar,
        deterministic,
        weight_out=0.1,
        device="cpu",
    ):
        self._input_size = observation_size + action_size
        self._output_size1 = reward_size
        self._output_size2 = observation_size

        self._max_logvar = max_logvar
        self._min_logvar = min_logvar

        self._hidden_layers = []
        fan_in = self._input_size
        self._parameter_number = 0
        for _ in range(hidden_layers):
            l = _BayesianLinerLayer(
                fan_in, hidden_layer_size, deterministic, weight_out, device=device
            )
            self._hidden_layers.append(l)
            self._parameter_number += l.parameter_number
            fan_in = hidden_layer_size

        self.output_layers = []
        self.output_layers.append(
            _BayesianLinerLayer(
                hidden_layer_size,
                reward_size * 2,
                deterministic,
                weight_out,
                device=device,
            )
        )
        self.output_layers.append(
            _BayesianLinerLayer(
                hidden_layer_size,
                observation_size * 2,
                deterministic,
                weight_out,
                device=device,
            )
        )

        self._parameter_number += self.output_layers[0].parameter_number
        self._parameter_number += self.output_layers[1].parameter_number
        self._distributional_parameter_number = self._parameter_number * 2

    @property
    def network_parameter_number(self):
        """The number elements in theta."""
        return self._parameter_number

    @property
    def distributional_parameter_number(self):
        """The number elements in phi."""
        return self._distributional_parameter_number

    def get_parameters(self):
        """Return mu and rho as a tuple of vectors."""
        params_mu, params_rho = zip(
            *[l.get_parameters() for l in self._hidden_layers + self.output_layers]
        )
        return torch.cat(params_mu), torch.cat(params_rho)

    def get_parameters_old(self):
        """Return mu and rho as a tuple of vectors."""
        params_mu, params_rho = zip(
            *[l.get_parameters_old() for l in self._hidden_layers + self.output_layers]
        )
        return torch.cat(params_mu), torch.cat(params_rho)

    def save_old_parameters(self):
        for l in self._hidden_layers + self.output_layers:
            l.save_old_params()

    def set_params(self, params_mu, params_rho):
        """Set a vector of parameters into weights and biases."""
        begin = 0
        for l in self._hidden_layers + self.output_layers:
            end = begin + l.parameter_number
            l.set_parameters(params_mu[begin:end], params_rho[begin:end])
            begin = end

    def infer(self, X, share_paremeters_among_samples=True):
        for layer in self._hidden_layers:
            X = F.elu(layer(X, share_paremeters_among_samples))

        X1 = self.output_layers[0](X, share_paremeters_among_samples)
        X2 = self.output_layers[1](X, share_paremeters_among_samples)

        mean1 = X1[:, : self._output_size1]
        logvar1 = X1[:, self._output_size1 :]
        logvar1 = torch.clamp(logvar1, min=self._min_logvar, max=self._max_logvar)

        mean2 = X2[:, : self._output_size2]
        logvar2 = X2[:, self._output_size2 :]
        logvar2 = torch.clamp(logvar2, min=self._min_logvar, max=self._max_logvar)

        return mean1, logvar1, mean2, logvar2

    def log_likelihood(self, input_batch, output_batch1, output_batch2):
        """Calculate an expectation of log likelihood.
        Mote Carlo approximation using a single parameter sample,
        i.e., E_{omega ~ q(* | phi)} [ log p(D | omega)] ~ log p(D | omega_1) ???
        """
        output_mean1, output_logvar1, output_mean2, output_logvar2 = self.infer(
            input_batch, share_paremeters_among_samples=True
        )

        ll_1 = -0.5 * (
            output_logvar1
            + (output_batch1 - output_mean1).pow(2) * (-output_logvar1).exp()
        ).sum(dim=1) - 0.5 * self._output_size1 * np.log(2 * np.pi)
        ll_2 = -0.5 * (
            output_logvar2
            + (output_batch2 - output_mean2).pow(2) * (-output_logvar2).exp()
        ).sum(dim=1) - 0.5 * self._output_size2 * np.log(2 * np.pi)

        obs_loss = (output_batch2 - output_mean2).pow(2).sum(dim=1).mean()
        r_loss = (output_batch1 - output_mean1).pow(2).sum(dim=1).mean()
        return 5 * ll_1.mean() + ll_2.mean(), obs_loss, r_loss
