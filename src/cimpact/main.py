""" 
Main method to handle the flow of causal impact analysis.
"""

import numpy as np
from cimpact.models.tensorflow_model import TensorFlowModel
from cimpact.models.prophet_model import ProphetModel
from cimpact.models.pyro_model import PyroModel
from cimpact.utils import (
    validate_data,
    regularize_time_series,
    convert_dates_to_indices,
    compute_p_value,
)


class CausalImpactAnalysis:
    """
    CausalImpactAnalysis class for running causal impact analysis using different models.

    Usage:
    ------
    # Define inputs: Model config, data, pre_period, post_period, covariates
    model_config = {
        'model_type': 'pyro',  # Options: 'prophet', 'tensorflow', 'pyro'
        'model_args': {
            'standardize': False,
            'learning_rate': 0.01,
            'num_iterations': 1000,
            'num_samples': 1000
        }
    }

    file_path = 'comparison_data.csv'
    index_col = 'DATE'  # Date column
    target_col = 'CHANGED'  # Target column

    data = pd.read_csv(file_path, index_col=index_col, parse_dates=True)
    pre_period = ['2019-04-16', '2019-07-14']
    post_period = ['2019-07-15', '2019-08-01']

    # Define color variables
    observed_color = "#000000"         # Black for observed
    predicted_color = "#7A00E6"        # Sanofi purple for predicted
    ci_color = "#D9B3FF66"             # Light lavender with transparency for CI
    intervention_color = "#444444"     # Dark gray for intervention
    figsize = (10,7)
    ci = 95                            # Confidence interval

    analysis = CausalImpactAnalysis(data,
                pre_period,
                post_period,
                model_config,
                index_col,
                target_col,
                observed_color, 
                predicted_color,
                ci_color, 
                intervention_color,
                figsize,
                ci
                )
    result = analysis.run_analysis()
    print(result)


    Available model arguments and their default values:

    TensorFlow Model:
    - standardize: True (Whether to standardize the data)
    - learning_rate: 0.01 (Learning rate for the optimizer)
    - num_variational_steps: 200 (Number of variational steps for Variational Inference)
    - num_results: 100 (Number of results for HMC sampling)
    - fit_method: 'vi' 
        - Fit method: 'vi' for Variational Inference,
        - 'hmc' for Hamiltonian Monte Carlo

    Prophet Model:
    - seasonality_mode: 'multiplicative' (Seasonality mode, either 'additive' or 'multiplicative')
    - yearly_seasonality: True (Whether to include yearly seasonality)
    - weekly_seasonality: True (Whether to include weekly seasonality)
    - daily_seasonality: False (Whether to include daily seasonality)
    - seasonality_prior_scale: 10.0 (Scale for seasonality prior)
    - changepoint_prior_scale: 0.05 (Scale for changepoint prior)

    Pyro Model:
    - standardize: False (Whether to standardize the data)
    - learning_rate: 0.01 (Learning rate for the optimizer)
    - num_iterations: 1000 (Number of iterations for SVI optimization)
    - num_samples: 1000 (Number of samples for posterior predictive sampling)
    """

    # add any new model class in this list
    models = {
        "tensorflow": TensorFlowModel,
        "prophet": ProphetModel,
        "pyro": PyroModel,
    }


    #pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(self, 
                data, 
                pre_period, 
                post_period, 
                config, 
                index_col,
                target_col, 
                observed_color="#000000", 
                predicted_color="#7A00E6",
                ci_color="#D9B3FF66", 
                intervention_color="#444444",
                figsize=(10, 7),
                ci=95):
        self.data = data
        self.pre_period = pre_period
        self.post_period = post_period
        self.config = config
        self.index_col = index_col
        self.target_col = target_col
        self.covariates = self.data[
            [col for col in data.columns if col not in [index_col, target_col]]
        ]
        self.model = None
        self.observed_color = observed_color
        self.predicted_color = predicted_color
        self.ci_color = ci_color
        self.intervention_color = intervention_color
        self.figsize = figsize
        self.ci = ci


    def calculate_zscore(self):
        """
        Function to calcualte z-score for confiedence interval
        
        The Z-score corresponds to the value in the standard normal distribution
        where the cumulative probability equals 1 − 𝛼/2, where α is the complement
        of the confidence level (e.g., for 95%, α=0.05).
        """
        # Convert confidence level to decimal (e.g., 95 -> 0.95)
        alpha = 1 - self.ci / 100
        # Calculate the cumulative probability for one tail
        cumulative_prob = 1 - (alpha / 2)
        # Use NumPy's percentile approximation for standard normal distribution
        z_score = np.percentile(np.random.normal(0, 1, 10**7), cumulative_prob * 100)
        return z_score

    def initialize_model(self):
        """
        Initialize the appropriate model based on the configuration.
        """
        model_type = self.config["model_type"]
        if  model_type in self.models:
            return self.models[model_type](
                self.data,
                self.pre_period,
                self.post_period,
                self.index_col,
                self.target_col,
                self.covariates,
                self.config["model_args"],
            ) # return the right model instantiated with necessary parameters
        else:
            raise ValueError(
                f"Model type '{model_type}' is not supported."
            )

    def run_analysis(self):
        """
        Run the causal impact analysis.
        """
        self.preprocess()
        self.model = self.initialize_model()
        self.model.fit()
        post_pred, pre_pred, combined_predictions, forecast_dist = self.model.predict()
        if post_pred is None or pre_pred is None or combined_predictions is None:
            raise ValueError("Prediction failed.")
        
        self.zscore = self.calculate_zscore()
        self.model.postprocess_results(post_pred, pre_pred, combined_predictions, self.zscore)
        summary = self.generate_summary(post_pred, forecast_dist)
        plot = self.model.plot(
                    combined_predictions,
                    observed_color=self.observed_color,       # Dodger blue for observed
                    predicted_color=self.predicted_color,      # Orange-red for predicted
                    ci_color=self.ci_color,  # Peach for confidence interval
                    intervention_color=self.intervention_color,    # Dark red for intervention
                    figsize=self.figsize,
                    zscore = self.zscore
                )
        return summary, plot

    def preprocess(self):
        """
        Preprocess the data.
        """
        self.data = self.data.ffill().bfill()
        self.data = regularize_time_series(self.data, date_col=self.index_col)
        if validate_data(self.data, self.pre_period, self.post_period):
            self.pre_period = convert_dates_to_indices(self.data, self.pre_period)
            self.post_period = convert_dates_to_indices(self.data, self.post_period)

    #pylint: disable=unused-argument
    def generate_summary(self, post_pred, forecast_dist):
        """
        Generate a summary of the causal impact analysis.
        """
        if "inferences" not in dir(self.model) or self.model.inferences is None:
            raise ValueError(
                "Inferences have not been computed. Run 'postprocess_results' first."
            )

        if self.config["model_type"] == "tensorflow":
            posterior_samples = forecast_dist.sample(
                self.config.get("num_results", 100)
            ).numpy()
            posterior_samples = np.array(posterior_samples)
            posterior_samples = np.squeeze(posterior_samples, axis=-1)
        elif self.config["model_type"] == "pyro":
            posterior_samples = forecast_dist["obs"]
        else:
            posterior_samples = forecast_dist["yhat"]

        tail_area_prob, causal_effect_prob = compute_p_value(
            posterior_samples, np.sum(self.model.post_data[self.target_col].values)
        )
        predicted_mean = self.model.inferences["predicted_mean"]
        ci_lower = self.model.inferences["ci_lower"]
        ci_upper = self.model.inferences["ci_upper"]
        actual_post = self.model.post_data[self.target_col].values

        abs_effect = actual_post - predicted_mean
        rel_effect = abs_effect / predicted_mean * 100
        cum_effect = np.cumsum(abs_effect)

        cumulative_rel_effect = np.sum(abs_effect) / np.sum(predicted_mean) * 100

        summary = f"""Summary results:

    Posterior inference {{CausalImpact}}

                            Average          Cumulative
    Actual                  {np.mean(actual_post):,.0f}            {np.sum(actual_post):,.0f}
    Prediction (s.d.)       {np.mean(predicted_mean):,.0f} (std {np.std(predicted_mean):,.0f})      {np.sum(predicted_mean):,.0f} ({np.std(predicted_mean):,.0f})
    {self.ci}% CI                  [{np.min(ci_lower):,.0f}, {np.max(ci_upper):,.0f}]   [{np.min(ci_lower):,.0f}, {np.max(ci_upper):,.0f}]

    Absolute effect (s.d.)  {np.mean(abs_effect):,.0f} (std {np.std(abs_effect):,.0f})       {np.sum(abs_effect):,.0f} (std {np.std(cum_effect):,.0f})
    {self.ci}% CI                  [{np.mean(abs_effect)-self.zscore*np.std(abs_effect):,.0f}, {np.mean(abs_effect)+self.zscore*np.std(abs_effect):,.0f}]            [{np.sum(abs_effect)-self.zscore*np.std(cum_effect):,.0f}, {np.sum(abs_effect)+self.zscore*np.std(cum_effect):,.0f}]

    Relative effect (s.d.)  {np.mean(rel_effect):.2f}% (std {np.std(rel_effect):.2f}%)   {cumulative_rel_effect:.2f}% (std {np.std(cum_effect)/np.sum(predicted_mean*1.)*100:.2f}%)
    {self.ci}% CI                  [{np.mean(rel_effect)-self.zscore*np.std(rel_effect):.2f}%, {np.mean(rel_effect)+self.zscore*np.std(rel_effect):.2f}%]            [{(np.sum(abs_effect)-self.zscore*np.std(cum_effect))/np.sum(predicted_mean*1.)*100:,.2f}%, {(np.sum(abs_effect)+self.zscore*np.std(cum_effect))/np.sum(predicted_mean*1.)*100:,.2f}%]          

    Posterior tail-area probability p: {tail_area_prob:.5f}
    Posterior probability of a causal effect: {causal_effect_prob:.2%}

    """

        return summary
