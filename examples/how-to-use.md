## Model Configurations

CImpact allows you to configure the underlying models to suit your analysis needs. Below are the configuration options for each supported model.

#### TensorFlow Model (Bayesian Structural Time Series)

Configure the TensorFlow model using the following parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `standardize` | `bool` | `True` | Whether to standardize the data before modeling. |
| `learning_rate` | `float` | `0.1` | Learning rate for the optimizer (used with Variational Inference). |
| `num_variational_steps` | `int` | `200` | Number of steps for variational inference. |
| `num_results` | `int` | `100` | Number of results for HMC sampling (used with HMC fit method). |
| `fit_method` | `str` | `'vi'` | Method for fitting the model. Options are `'vi'` (Variational Inference) and `'hmc'` (Hamiltonian Monte Carlo). |

**Example Configuration:**

```python
model_config = {
    'model_type': 'tensorflow',
    'model_args': {
        'standardize': True,
        'learning_rate': 0.1,
        'num_variational_steps': 200,
        'fit_method': 'vi'
    }
}
```

#### Prophet Model

Configure the Prophet model with the following parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `seasonality_mode` | `str` | `'multiplicative'` | Seasonality mode, either `'additive'` or `'multiplicative'`. |
| `yearly_seasonality` | `bool` | `True` | Whether to include yearly seasonality. |
| `weekly_seasonality` | `bool` | `True` | Whether to include weekly seasonality. |
| `daily_seasonality` | `bool` | `False` | Whether to include daily seasonality. |
| `seasonality_prior_scale` | `float` | `10.0` | Scale for seasonality prior. |
| `changepoint_prior_scale` | `float` | `0.05` | Scale for changepoint prior. |
| Additional Parameters | - | - | Pass any additional parameters supported by Prophet (e.g., `holidays`). |

**Example Configuration:**

```python
model_config = {
    'model_type': 'prophet',
    'model_args': {
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'seasonality_prior_scale': 10.0,
        'changepoint_prior_scale': 0.05,
        'holidays': your_holidays_dataframe  # Optional
    }
}
```

#### Pyro Model

Configure the Pyro model using the following parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `standardize` | `bool` | `False` | Whether to standardize the data before modeling. |
| `learning_rate` | `float` | `0.01` | Learning rate for the optimizer (maps to `lr` for Adam optimizer). |
| `num_iterations` | `int` | `1000` | Number of iterations for SVI optimization. |
| `num_samples` | `int` | `1000` | Number of samples for posterior predictive sampling. |

**Example Configuration:**

```python
model_config = {
    'model_type': 'pyro',
    'model_args': {
        'standardize': False,
        'learning_rate': 0.01,
        'num_iterations': 1000,
        'num_samples': 1000
    }
}
```

## Example Usage

#### Pyro model

```python

import pandas as pd
from cimpact import CausalImpactAnalysis

# Load your data
data = pd.read_csv('https://raw.githubusercontent.com/Sanofi-Public/CImpact/master/examples/google_data.csv')

# Define the configuration for the model
model_config = {
    'model_type': 'pyro',
    'model_args': {
        'standardize': False,
        'learning_rate': 0.01,
        'num_iterations': 1000,
        'num_samples': 1000
    }
}

# Define the pre and post-intervention periods
pre_period = ['2020-01-01', '2020-03-13']
post_period = ['2020-03-14', '2020-03-31']

#Define index column and target column
index_col = 'date'
target_col = 'y'

# Define color variables
observed_color = "#000000"         # Black for observed
predicted_color = "#7A00E6"        # Sanofi purple for predicted
ci_color = "#D9B3FF66"             # Light lavender with transparency for CI
intervention_color = "#444444"     # Dark gray for intervention
figsize = (10,7)
ci = 95                            # Confidence interval

# Run the analysis
analysis = CausalImpactAnalysis(data, pre_period, post_period, model_config, index_col, target_col, observed_color,  predicted_color, ci_color, intervention_color, figsize, ci)
result = analysis.run_analysis()
print(result)
```

##### Outcome

![Result visualization for Tensorflow model](https://github.com/Sanofi-Public/CImpact/blob/master/examples/results/pyro_google_data_results.png "Result visualization for Tensorflow model")

Summary results:

    Posterior inference {CausalImpact}

                            Average          Cumulative
    Actual                  145              2,614
    Prediction (s.d.)       130 (std 718)    2,348 (718)
    95% CI                  [-2,485, 2,853]  [-2,485, 2,853]

    Absolute effect (s.d.)  15 (std 717)     266 (std 717)
    95% CI                  [-1,315, 1,211]  [-1,315, 1,211]

    Relative effect (s.d.)  -73.12% (std 80.40%)   -1316.23% (std 80.40%)
    95% CI                  [-161.97%, 192.83%]    [-161.97%, 192.83%]

    Posterior tail-area probability p: 0.33167
    Posterior probability of a causal effect: 66.83%
    
    Model Performance Metrics:
    RMSE: 25.45
    MAPE: 18.32%

***Note:** As you can see here, not all models will result into good results! You need to finetune model config to get the best possible result with the model. 

#### Prophet model

```python

import pandas as pd
from cimpact import CausalImpactAnalysis

# Load your data
data = pd.read_csv('https://raw.githubusercontent.com/Sanofi-Public/CImpact/master/examples/google_data.csv')

# Define the configuration for the model
model_config = {
    'model_type': 'prophet',
    'model_args': {
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'seasonality_prior_scale': 10.0,
        'changepoint_prior_scale': 0.05
    }
}

# Define the pre and post-intervention periods
pre_period = ['2020-01-01', '2020-03-13']
post_period = ['2020-03-14', '2020-03-31']

#Define index column and target column
index_col = 'date'
target_col = 'y'

# Define color variables
observed_color = "#000000"         # Black for observed
predicted_color = "#7A00E6"        # Sanofi purple for predicted
ci_color = "#D9B3FF66"             # Light lavender with transparency for CI
intervention_color = "#444444"     # Dark gray for intervention
figsize = (10,7)
ci = 95                            # Confidence interval

# Run the analysis
analysis = CausalImpactAnalysis(data, pre_period, post_period, model_config, index_col, target_col, observed_color,  predicted_color, ci_color, intervention_color, figsize, ci)
result = analysis.run_analysis()
print(result)
```

##### Outcome

![Result visualization for Tensorflow model](https://github.com/Sanofi-Public/CImpact/blob/master/examples/results/prophet_google_data_results.png "Result visualization for Tensorflow model")

Summary results:

    Posterior inference {CausalImpact}

                            Average          Cumulative
    Actual                  145              2,614
    Prediction (s.d.)       170 (std 7)      3,064 (7)
    95% CI                  [142, 195]       [142, 195]

    Absolute effect (s.d.)  -25 (std 14)     -450 (std 14)
    95% CI                  [-49, 2]         [-49, 2]

    Relative effect (s.d.)  -14.55% (std 8.08%)   -261.88% (std 8.08%)
    95% CI                  [-28.03%, 1.38%]      [-28.03%, 1.38%]

    Posterior tail-area probability p: 0.00000
    Posterior probability of a causal effect: 100.00%
    
    Model Performance Metrics:
    RMSE: 12.34
    MAPE: 8.56%