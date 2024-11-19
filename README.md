[![License](https://img.shields.io/badge/License-Academic%20Non--Commercial-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Prophet](https://img.shields.io/badge/Prophet-1.x-blueviolet.svg)](https://facebook.github.io/prophet/)
[![Pyro](https://img.shields.io/badge/Pyro-1.x-brightgreen.svg)](https://pyro.ai/)

<img src="https://github.com/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/main/assets/logo.png" width=70% height=40%>

CImpact - Causal Inference for Measuring Performance and Causal Trends
======================================================================



[](LICENSE)

CImpact is a modular causal impact analysis library for Python, supporting multiple time series models, including [TensorFlow](https://www.tensorflow.org/probability/overview) , [Prophet](https://facebook.github.io/prophet/), and [Pyro](https://pyro.ai/examples/). It provides a flexible framework for estimating the causal effect of an intervention on time series data.

Table of Contents
-----------------

-   [Introduction](#introduction)
-   [Features](#features)
-   [Why CImpact?](#why-cimpact)
-   [Code Structure](#code-structure)
-   [Installation](#installation)
-   [Getting Started](#getting-started)
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


## Why CImpact?
------------

CImpact extends the functionalities of the [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) library by incorporating support for multiple modeling approaches. This modular design allows users to choose the best model for their specific needs and compare performance and results across different models. We highly recommend reading this detailed [blog post](https://towardsdatascience.com/implementing-causal-impact-on-top-of-tensorflow-probability-c837ea18b126) explainng the causal inference in great detail.

## Features
--------

-   **Support for Multiple Models**: Utilize TensorFlow, Prophet, or Pyro models according to your needs.
-   **Modular Design**: Easily extend the library with new models due to its adapter-based architecture.
-   **Flexible Configuration**: Customize model settings and hyperparameters to suit specific analysis requirements.
-   **Comprehensive Evaluation**: Integrated methods for assessing model performance and the causal impact of interventions.
-   **Enhanced Visualization**: Generate insightful plots for better interpretation of results.

## Code Structure
------------

```plaintext
CImpact/
├── .checkmarx/scan_results       # Contains results from code scanning for security and quality issues
├── .github/                      # GitHub configuration files for workflows and actions
├── assets/                       # Stores media assets, such as the project logo, used in the README or documentation
├── examples/                     # Example scripts showcasing usage of the library and sample data for testing
├── scripts/                      # Utility scripts for code cleaning, formatting, and other maintenance tasks
├── src/                          # Core library source code, including main modules and adapters for different models
├── tests/                        # Test cases for ensuring code functionality and correctness across modules
├── .coveragerc                   # Configuration file for coverage reporting, specifying which files to include/exclude
├── .gitignore                    # Specifies files and directories for Git to ignore
├── .pylintrc                     # Configuration for Python linter (Pylint) to enforce code style and quality standards
├── CONTRIBUTING.md               # Guidelines for contributing to the project
├── LICENSE.txt                   # License information for the project, detailing usage rights and limitations
├── Makefile                      # Commands for building, testing, and packaging the project in a standard way
├── README.md                     # Project introduction, usage instructions, and documentation (this file)
├── __init__.py                   # Marks the directory as a Python package
├── pyproject.toml                # Python packaging configuration file for managing dependencies and metadata
├── requirements.txt              # List of Python dependencies required to run the project
```

## Installation
------------

CImpact can be installed using one of the following methods:

### 1. Stable Release (Coming Soon)

The stable release of CImpact will soon be available on PyPI. Once published, you can install it with:

```bash
pip install cimpact
```

Stay tuned for updates on the stable release!

### 2. Latest Release (Manual Installation)

To access the latest features or contribute to development, you can manually install CImpact by building it from source. Follow the steps below:

**Step 1: Clone the Repository**

Clone the CImpact repository to your local machine:

```bash
git clone https://github.com/Sanofi-OneAI/oneai-com-turing-causal_inference.git
cd oneai-com-turing-causal_inference
```

**Step 2: Install Dependencies** 

Install the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Step 3: Build the Wheel File** 

Build the library into a Python Wheel file:

```bash
python -m build
```

The generated `.whl` file will be located in the `dist/` directory.

**Step 4: Install the Wheel File** 

Use `pip` to install the wheel file:

```bash
pip install dist/cimpact-<version>.whl
```

Replace `<version>` with the version number of the generated `.whl` file. This will install the cimpact library in your environment and now you can use it using the following steps. 


## Getting Started
---------------

### Example Usage

#### Tensowflow model

```python

import pandas as pd
from cimpact import CausalImpactAnalysis

# Load your data
data = pd.read_csv('https://github.com/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/main/examples/google_data.csv')

# Define the configuration for the model
model_config = {
    'model_type': 'tensorflow',  # Options: 'tensorflow', 'prophet', 'pyro'
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

# Define color variables (optional arguments)
observed_color = "#000000"         # Black for observed
predicted_color = "#7A00E6"        # Sanofi purple for predicted
ci_color = "#D9B3FF66"             # Light lavender with transparency for CI
intervention_color = "#444444"     # Dark gray for intervention
figsize = (10,7)

# Run the analysis
analysis = CausalImpactAnalysis(data, pre_period, post_period, model_config, index_col, target_col, observed_color,  predicted_color, ci_color, intervention_color)
result = analysis.run_analysis()
print(result)
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


> [!NOTE]  
> Please refer to `examples/how-to-use.md` for detailed model configuration instructions and additional usage examples of the library.

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

We welcome contributions to enhance and refine the library. While we are particularly interested in contributions in the following areas, we are open to other suggestions as well. If you have any ideas, please create an issue to discuss potential contributions.

- Add new, qualified models to broaden analytical options. We are currently exploring zero-shot learning models like [Google timesfm](https://github.com/google-research/timesfm) or [Amazon Chronos](https://www.amazon.science/code-and-datasets/chronos-learning-the-language-of-time-series).
- Enhanced Visualization**: Develop advanced plotting functions for deeper insights and a better understanding of results.
- Publish detailed tutorials to help users in effectively utilizing the library.

## Contributing
------------

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to participate.

## License
-------

This work is available for academic research and non-commercial use only. See the <LICENSE> file for details.

## Acknowledgements
----------------
We are thankful of Google research (cited below[^1]) team for publishing **"Inferring causal impact using Bayesian structural time-series models"** research paper and sharing orginal [R package](https://github.com/google/CausalImpact) to open souce community. We also extend our gratitude to the authors of [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) for their foundational work, which inspired this library.  


[^1]: Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015). Inferring causal impact using Bayesian structural time-series models.

* * * * *
