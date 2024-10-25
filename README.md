[![License](https://img.shields.io/badge/License-Academic%20Non--Commercial-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Prophet](https://img.shields.io/badge/Prophet-1.x-blueviolet.svg)](https://facebook.github.io/prophet/)
[![Pyro](https://img.shields.io/badge/Pyro-1.x-brightgreen.svg)](https://pyro.ai/)

CImpact - Causal Inference for Measuring Performance and Causal Trends
======================================================================



[](LICENSE)

CImpact is a modular causal impact analysis library for Python, supporting multiple time series models, including TensorFlow, Prophet, and Pyro. It provides a flexible framework for estimating the causal effect of an intervention on time series data.

Table of Contents
-----------------

-   [Introduction](#introduction)
-   [Features](#features)
-   [Why CImpact?](#why-cimpact)
-   [Installation](#installation)
-   [Getting Started](#getting-started)
    -   [Model Configurations](#model-configurations)
    -   [Example Usage](#example-usage)
-   [Evaluation Methods](#evaluation-methods)
-   [Performance Comparison](#performance-comparison)
-   [Future Plans](#future-plans)
-   [Contributing](#contributing)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

## Introduction
------------

CImpact is designed to help analysts and data scientists assess the impact of an intervention on time series data. By leveraging different statistical models, CImpact aims to provide robust causal inference results, accommodating various use cases and preferences in model selection.

## Features
--------

-   **Support for Multiple Models**: Utilize TensorFlow, Prophet, or Pyro models according to your needs.
-   **Modular Design**: Easily extend the library with new models due to its adapter-based architecture.
-   **Flexible Configuration**: Customize model settings and hyperparameters to suit specific analysis requirements.
-   **Comprehensive Evaluation**: Integrated methods for assessing model performance and the causal impact of interventions.
-   **Enhanced Visualization**: Generate insightful plots for better interpretation of results.

## Why CImpact?
------------

CImpact extends the functionalities of the [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) library by incorporating support for multiple modeling approaches. This modular design allows users to choose the best model for their specific needs and compare performance and results across different models.

## Installation
------------

Install CImpact using `pip`:

bash

Copy code

`pip install cimpact`


## Getting Started
---------------

## Model Configurations

CImpact allows you to configure the underlying models to suit your analysis needs. Below are the configuration options for each supported model.

#### TensorFlow Model (Bayesian Structural Time Series)

Configure the TensorFlow model using the following parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `standardize` | `bool` | `True` | Whether to standardize the data before modeling. |
| `learning_rate` | `float` | `0.01` | Learning rate for the optimizer. |
| `num_variational_steps` | `int` | `1000` | Number of steps for variational inference. |
| `fit_method` | `str` | `'vi'` | Method for fitting the model. Options are `'vi'` (Variational Inference) and `'hmc'` (Hamiltonian Monte Carlo). |

**Example Configuration:**

```python
model_config = {
    'model_type': 'tensorflow',
    'model_args': {
        'standardize': True,
        'learning_rate': 0.01,
        'num_variational_steps': 1000,
        'fit_method': 'vi'
    }
}
```

#### Prophet Model

Configure the Prophet model with the following parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `standardize` | `bool` | `True` | Whether to standardize the data before modeling. |
| Additional Parameters | - | - | Pass any additional parameters supported by Prophet (e.g., `seasonality_mode`, `holidays`). |

**Example Configuration:**

```python
model_config = {
    'model_type': 'prophet',
    'model_args': {
        'standardize': True,
        'seasonality_mode': 'multiplicative',
        'weekly_seasonality': True,
        'holidays': your_holidays_dataframe
    }
}
```

#### Pyro Model

Configure the Pyro model using the following parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `standardize` | `bool` | `True` | Whether to standardize the data before modeling. |
| `learning_rate` | `float` | `0.01` | Learning rate for the optimizer. |
| `num_iterations` | `int` | `1000` | Number of iterations for training. |
| `num_samples` | `int` | `1000` | Number of samples to draw for prediction. |

**Example Configuration:**

```python
model_config = {
    'model_type': 'pyro',
    'model_args': {
        'standardize': True,
        'learning_rate': 0.01,
        'num_iterations': 1000,
        'num_samples': 1000
    }
}
```

### Example Usage

#### Tensowflow model

```python

import pandas as pd
from cimpact import CausalImpactAnalysis

# Load your data
data = pd.read_csv('https://github.com/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/main/examples/google_data.csv')

# Define the configuration for the model
model_config = {
    'model_type': 'pyro',  # Options: 'tensorflow', 'prophet', 'pyro'
    'model_args': {
        'standardize': True,
        'learning_rate': 0.01,
        'num_variational_steps': 1000,
        'fit_method': 'vi'
    }
}

# Define the pre and post-intervention periods
pre_period = ['2020-01-01', '2020-03-13']
post_period = ['2020-03-14', '2020-03-31']

#Define index column and target column
index_col = 'date'
target_col = 'y'

# Run the analysis
analysis = CausalImpactAnalysis(
            data,
            pre_period,
            post_period,
            model_config,
            index_col,
            target_col,
        )

result, plot = analysis.run_analysis()
print(result.summary())
result.plot()
```

##### Outcome

![Result visualization for Tensorflow model](https://github.com/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/main/examples/results/tensorflow_google_data_results.png "Result visualization for Tensorflow model")

Posterior inference {CIMpact}

|                                       | Average          | Cumulative       |
|---------------------------------------|------------------|------------------|
| **Actual**                            | 145             | 2,614           |
| **Prediction (s.d.)**                 | 180 (10)        | 3,237 (10)      |
| **95% CI**                            | [144, 218]      | [2,880, 3,594]  |
| **Absolute effect (s.d.)**            | -35 (15)        | -623 (15)       |
| **95% CI**                            | [-61, -11]      | [-980, -266]    |
| **Relative effect (s.d.)**            | -19.08% (7.58%) | -19.08% (7.58%) |
| **95% CI**                            | [-32.42%, -6.66%] | [-32.42%, -6.66%] |
| **Posterior tail-area probability p:** | 0.15842        |                  |
| **Posterior probability of a causal effect:** | 84.16%      |                  |


#### Pyro model

```python

import pandas as pd
from cimpact import CausalImpactAnalysis

# Load your data
data = pd.read_csv('https://github.com/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/main/examples/google_data.csv')

# Define the configuration for the model
model_config = {
    'model_type': 'pyro',
    'model_args': {
        'standardize': True,
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

# Run the analysis
analysis = CausalImpactAnalysis(
            data,
            pre_period,
            post_period,
            model_config,
            index_col,
            target_col,
        )

result, plot = analysis.run_analysis()
print(result.summary())
result.plot()
```

##### Outcome

![Result visualization for Tensorflow model](https://github.com/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/main/examples/results/pyro_google_data_results.png "Result visualization for Tensorflow model")

Posterior inference {CIMpact}

|                                       | Average               | Cumulative         |
|---------------------------------------|-----------------------|--------------------|
| **Actual**                            | 145                  | 2,614             |
| **Prediction (s.d.)**                 | 130 (718)            | 2,348 (718)       |
| **95% CI**                            | [-2,485, 2,853]      | [-2,485, 2,853]   |
| **Absolute effect (s.d.)**            | 15 (717)             | 266 (717)         |
| **95% CI**                            | [-1,315, 1,211]      | [-1,315, 1,211]   |
| **Relative effect (s.d.)**            | -73.12% (80.40%)     | -1316.23% (80.40%) |
| **95% CI**                            | [-161.97%, 192.83%]  | [-161.97%, 192.83%] |
| **Posterior tail-area probability p:** | 0.33167             |                    |
| **Posterior probability of a causal effect:** | 66.83%           |                    |

***Note:** As you can see here, not all models will result into good results! You need to finetune model config to get the best possible result with the model. 

#### Prophet model

```python

import pandas as pd
from cimpact import CausalImpactAnalysis

# Load your data
data = pd.read_csv('https://github.com/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/main/examples/google_data.csv')

# Define the configuration for the model
model_config = {
    'model_type': 'prophet',
    'model_args': {
        'standardize': True,
        'learning_rate': 0.01,
        'num_variational_steps': 1000,
        'weekly_seasonality': False,
    }
}

# Define the pre and post-intervention periods
pre_period = ['2020-01-01', '2020-03-13']
post_period = ['2020-03-14', '2020-03-31']

#Define index column and target column
index_col = 'date'
target_col = 'y'

# Run the analysis
analysis = CausalImpactAnalysis(
            data,
            pre_period,
            post_period,
            model_config,
            index_col,
            target_col,
        )

result, plot = analysis.run_analysis()
print(result.summary())
result.plot()
```
##### Outcome

![Result visualization for Tensorflow model](https://github.com/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/main/examples/results/prophet_google_data_results.png "Result visualization for Tensorflow model")

Posterior inference {CIMpact}

|                                       | Average             | Cumulative       |
|---------------------------------------|---------------------|------------------|
| **Actual**                            | 145                | 2,614           |
| **Prediction (s.d.)**                 | 170 (7)            | 3,064 (7)       |
| **95% CI**                            | [142, 195]         | [142, 195]      |
| **Absolute effect (s.d.)**            | -25 (14)           | -450 (14)       |
| **95% CI**                            | [-49, 2]           | [-49, 2]        |
| **Relative effect (s.d.)**            | -14.55% (8.08%)    | -261.88% (8.08%) |
| **95% CI**                            | [-28.03%, 1.38%]   | [-28.03%, 1.38%] |
| **Posterior tail-area probability p:** | 0.00000           |                  |
| **Posterior probability of a causal effect:** | 100.00%      |                  |


## Evaluation Methods
------------------

CImpact includes comprehensive evaluation methods to assess model performance and the causal impact of interventions:

-   **Summary Statistics**: Provides point estimates and confidence intervals for the estimated impact.
-   **Visualization**: Plots observed data, counterfactual predictions, and estimated impact over time.
-   **Diagnostics**: Offers residual analysis and model diagnostics to assess fit.

## Performance Comparison
----------------------

Our performance comparisons highlight:

-   **TensorFlow**: Robust performance with flexibility for advanced inference methods like variational inference and HMC.
-   **Prophet**: User-friendly with built-in seasonality and holiday effects; may be slower with large datasets.
-   **Pyro**: Strong Bayesian inference capabilities; may require more computational resources.

## Future Plans
------------

-   **Expand Model Support**: Incorporate additional models to broaden analytical options.
-   **Enhanced Visualization**: Develop advanced plotting functions for deeper insights.
-   **Documentation**: Provide detailed documentation and tutorials.
-   **Community Engagement**: Encourage contributions to extend and refine the library.

## Contributing
------------

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to participate.

## License
-------

This work is available for academic research and non-commercial use only. See the <LICENSE> file for details.

## Acknowledgements
----------------

We extend our gratitude to the authors of [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) for their foundational work, which inspired this library.

* * * * *