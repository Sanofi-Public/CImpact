"""
Module for TensorFlow-based Bayesian Structural Time Series model implementation.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from cimpact.models.base_model import BaseModel


class TensorFlowModel(BaseModel):
    """
    Modeling class for TensorFlow's Bayesian Structural Time Series model, extending the Base Model.
    This class provides methods to fit the model, make predictions, and evaluate model performance.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
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
        self.pre_data = None
        self.post_data = None
        self.variational_posteriors = None

        self.covariates = covariates
        self.model = self.build_model()
        self.samples = None
        self.inferences = None

    def build_model(self):
        """
        Build the Bayesian Structural Time Series model.
        """
        observed_time_series = self.data[self.target_col][: self.pre_period[1] + 1]
        design_matrix = self.covariates[: self.pre_period[1] + 1].values

        local_level = tfp.sts.LocalLinearTrend(
            observed_time_series=observed_time_series
        )
        seasonal = tfp.sts.Seasonal(
            num_seasons=12, observed_time_series=observed_time_series
        )
        regression = tfp.sts.LinearRegression(design_matrix=design_matrix)

        model = tfp.sts.Sum(
            [local_level, seasonal, regression],
            observed_time_series=observed_time_series,
        )
        return model

    def fit(self):
        """
        Fit the TensorFlow model using the pre-intervention data.
        """
        data, pre_data, post_data = self.preprocess_data()
        self.data = data
        self.pre_data = pre_data
        self.post_data = post_data
        self.model = self.initialize_model(pre_data[self.target_col])
        fit_method = self.model_args.get("fit_method", "vi")

        try:
            if fit_method == "hmc":
                self.samples, _ = self.fit_with_hmc(pre_data[self.target_col])
            elif fit_method == "vi":
                self.samples, elbo_loss_curve = self.fit_with_vi(
                    pre_data[self.target_col]
                )
            else:
                raise ValueError(f"Unsupported fit method: {fit_method}")
            print("Model fitting completed successfully.")
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"An error occurred during model fitting: {str(e)}")
            self.variational_posteriors = (
                None  # Reset to ensure predict doesn't proceed with incomplete fitting
            )

    def fit_with_hmc(self, observed_time_series):
        """
        Fit the model using Hamiltonian Monte Carlo (HMC).
        """
        num_results = self.model_args.get("num_results", 100)
        samples, kernel_results = tfp.sts.fit_with_hmc(
            model=self.model,
            observed_time_series=observed_time_series,
            num_results=num_results,
        )
        return samples, kernel_results

    #pylint: disable=no-member
    def fit_with_vi(self, observed_time_series):
        """
        Fit the model using Variational Inference (VI).
        """
        learning_rate = self.model_args.get("learning_rate", 0.1)
        num_variational_steps = self.model_args.get("num_variational_steps", 200)

        optimizer = tf.optimizers.Adam(learning_rate)
        self.variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
            model=self.model
        )

        @tf.function
        def _run_vi():
            elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=self.model.joint_log_prob(observed_time_series),
                surrogate_posterior=self.variational_posteriors,
                optimizer=optimizer,
                num_steps=num_variational_steps,
            )
            samples = self.variational_posteriors.sample(50)  # Assumption: 50 samples
            return samples, elbo_loss_curve

        return _run_vi()

    def predict(self):
        """
        Make predictions using the TensorFlow model.

        Returns:
        - forecast (np.array): Forecasted values.
        - pre_pred (np.array): Predictions for the pre-intervention period.
        - combined_predictions (np.array): Combined predictions for the full period.
        - forecast_dist (tfp.distributions.Distribution): Forecast distribution.
        """
        if self.samples is not None:
            forecast_dist = tfp.sts.forecast(
                model=self.model,
                observed_time_series=self.data[self.target_col][
                    self.pre_period[0]: self.pre_period[1] + 1
                ],  # Use pre-period data
                parameter_samples=self.samples,
                num_steps_forecast=len(
                    self.post_data
                ),  # Adjusted to include only post period length
            )
        else:
            forecast_dist = tfp.sts.forecast(
                model=self.model,
                observed_time_series=self.pre_data,
                parameter_samples=self.variational_posteriors.sample(50),
                num_steps_forecast=len(self.data)
                - self.pre_period[1]
                - 1,  # Adjusted to include only post period length
            )

        one_step_dist = tfp.sts.one_step_predictive(
            model=self.model,
            observed_time_series=self.pre_data[
                self.target_col
            ],  # Only pre-intervention data
            parameter_samples=self.samples,
        )

        forecast = forecast_dist.mean().numpy()
        if len(forecast.shape) == 1 or forecast.shape[1] == 1:
            forecast = forecast.ravel()  # Flatten array if it's 2D but only 1 column
        pre_pred = np.squeeze(
            one_step_dist.sample(self.model_args.get("num_results", 100)).numpy()
        )

        # Ensure pre_pred is 2D for concatenation
        if pre_pred.ndim == 1:
            pre_pred = np.expand_dims(pre_pred, axis=1)

        combined_predictions = np.concatenate([pre_pred.mean(axis=0), forecast])
        return forecast, pre_pred, combined_predictions, forecast_dist

    def initialize_model(self, observed_time_series):
        """
        Initialize the Structural Time Series model components.
        """
        local_level = tfp.sts.LocalLevel(
            observed_time_series=observed_time_series, name="local_level"
        )
        seasonal = tfp.sts.Seasonal(
            num_seasons=12, observed_time_series=observed_time_series, name="seasonal"
        )
        trend = tfp.sts.LocalLinearTrend(
            observed_time_series=observed_time_series, name="trend"
        )

        design_matrix = self.covariates.values.astype(np.float64)
        regression = tfp.sts.SparseLinearRegression(
            design_matrix=design_matrix, name="regression"
        )

        model = tfp.sts.Sum(
            [local_level, seasonal, trend, regression],
            observed_time_series=observed_time_series,
        )
        return model
