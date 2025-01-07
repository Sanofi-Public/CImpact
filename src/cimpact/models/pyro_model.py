""" 
Pyro model for causal impact measurement
"""

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from pyro import optim
from pyro.infer import SVI, Trace_ELBO, Predictive
from cimpact.models.base_model import BaseModel


#pylint: disable=abstract-method
class BayesianRegressionModel(nn.Module):
    """
    Bayesian Regression Model using Pyro and PyTorch.
    This model defines the probabilistic model and guide for variational inference.
    """

    def __init__(self, input_dim):
        """ 
        Constructor method for Bayesian Regression Model
        """
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)
        self.pre_data = None
        self.post_data = None

    def model(self, x_data, y_data=None):
        """ 
        Model method for Bayesian Regression Model
        """
        beta = pyro.sample(
            "beta",
            dist.Normal(
                torch.zeros(self.input_dim), torch.ones(self.input_dim)
            ).to_event(1),
        )
        sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(1.0)))
        mean = (x_data * beta).sum(-1)
        with pyro.plate("data", len(x_data)):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=y_data)

    def guide(self, x_data, y_data=None): #pylint: disable=unused-argument
        """ 
        Guide method for Bayesian Regression Model
        """
        beta_loc = pyro.param("beta_loc", torch.zeros(self.input_dim))
        beta_scale = pyro.param(
            "beta_scale",
            torch.ones(self.input_dim),
            constraint=dist.constraints.positive,
        )
        sigma_loc = pyro.param(
            "sigma_loc", torch.tensor(1.0), constraint=dist.constraints.positive
        )
        pyro.sample("beta", dist.Normal(beta_loc, beta_scale).to_event(1))
        pyro.sample("sigma", dist.HalfCauchy(sigma_loc))


class PyroModel(BaseModel):
    """
    Modeling class for the Pyro Bayesian regression model, extending the Base Model.
    This class provides methods to fit the model, make predictions, and evaluate model performance.
    """

    #pylint: disable=too-many-arguments, no-member
    def __init__(
        self,
        data,
        pre_period,
        post_period,
        index_col,
        target_col,
        covariates,
        model_args,
    ):
        super().__init__(
            data, pre_period, post_period, index_col, target_col, model_args
        )
        self.covariates = covariates
        self.model = BayesianRegressionModel(input_dim=self.covariates.shape[1])
        self.optimizer = optim.Adam({"lr": self.model_args.get("learning_rate", 0.01)})
        self.svi = SVI(
            self.model.model, self.model.guide, self.optimizer, loss=Trace_ELBO()
        )
        self.pre_data = None

    #pylint: disable=unused-variable
    def fit(self):
        """
        Fit the Pyro model using the pre-intervention data.
        """
        data, pre_data, post_data = self.preprocess_data()
        self.data = data
        self.pre_data = pre_data
        self.post_data = post_data

        train_data = torch.tensor(pre_data[self.target_col].values, dtype=torch.float)
        covariates = torch.tensor(
            pre_data[self.covariates.columns].values, dtype=torch.float
        )

        num_iterations = self.model_args.get("num_iterations", 1000)
        for _ in range(num_iterations):
            loss = self.svi.step(covariates, train_data)

    def predict(self):
        """
        Make predictions using the Pyro model.

        Returns:
        - post_pred (np.array): Predictions for the post-intervention period.
        - pre_pred (np.array): Predictions for the pre-intervention period.
        - combined_predictions (np.array): Combined predictions for the full period.
        - samples (dict): Samples from the predictive distribution.
        """
        covariates = torch.tensor(
            self.data[self.covariates.columns].values, dtype=torch.float
        )
        predictive = Predictive(
            self.model.model, guide=self.model.guide, num_samples=1000
        )
        samples = predictive(covariates)

        forecast = samples["obs"].mean(axis=0).detach().numpy()
        pre_pred = forecast[self.pre_period[0]: self.pre_period[1] + 1]
        post_pred = forecast[self.pre_period[1] + 1 :]
        combined_predictions = np.concatenate([pre_pred, post_pred])

        return post_pred, pre_pred, combined_predictions, samples
