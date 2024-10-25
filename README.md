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
    -   [Example Usage](#example-usage)
    -   [Model Configurations](#model-configurations)
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

### Example Usage

```python

import pandas as pd
from cimpact import CausalImpactAnalysis

# Load your data
data = pd.read_csv('your_data.csv', parse_dates=['DATE'], index_col='DATE')

# Define the configuration for the model
model_config = {
    'model_type': 'pyro',  # Options: 'tensorflow', 'prophet', 'pyro'
    'model_args': {
        'standardize': True,
        'learning_rate': 0.01,
        'num_iterations': 1000,
        'num_samples': 1000
    }
}

# Define the pre and post-intervention periods
pre_period = ['2020-01-01', '2020-06-01']
post_period = ['2020-06-02', '2020-12-31']

# Run the analysis
analysis = CausalImpactAnalysis(
    data=data,
    pre_period=pre_period,
    post_period=post_period,
    model_config=model_config,
    date_col='DATE',
    target_col='TARGET'
)

result = analysis.run_analysis()
print(result.summary())
result.plot()
```

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
