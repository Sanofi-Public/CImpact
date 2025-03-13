[![License](https://img.shields.io/badge/License-Academic%20Non--Commercial-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Prophet](https://img.shields.io/badge/Prophet-1.x-blueviolet.svg)](https://facebook.github.io/prophet/)
[![Pyro](https://img.shields.io/badge/Pyro-1.x-brightgreen.svg)](https://pyro.ai/)

<img src="https://github.com/Sanofi-Public/CImpact/blob/master/assets/logo.png" width=70% height=40%>

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

CImpact is a versatile Python library designed to empower analysts and data scientists to evaluate the causal impact of interventions on time series data. By integrating a suite of statistical and probabilistic models, CImpact offers robust and flexible tools for causal inference, ensuring adaptability across diverse use cases and modeling preferences.

With support for multiple cutting-edge frameworks, including TensorFlow Probability, Pyro, and Prophet, CImpact enables users to:

Quantify Intervention Effects: Measure the influence of interventions with confidence intervals and probabilistic predictions.
Leverage Advanced Models: Utilize models that capture trends, seasonality, and covariates, providing deeper insights into time series dynamics.
Customize Approaches: Select between Hamiltonian Monte Carlo (HMC), Variational Inference (VI), or Prophet-based methods to match computational and analytical needs.
Seamlessly Handle Covariates: Account for external variables that impact the time series through regression components or regressors.

Whether your data exhibits complex seasonality, local trends, or requires the incorporation of contextual variables, CImpact equips you with a powerful toolkit to make informed decisions supported by rigorous statistical analysis.

## Why CImpact?
------------

CImpact extends the functionalities of the [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) library by incorporating support for multiple modeling approaches. This modular design allows users to choose the best model for their specific needs and compare performance and results across different models. We highly recommend reading this detailed [blog post](https://towardsdatascience.com/implementing-causal-impact-on-top-of-tensorflow-probability-c837ea18b126) explainng the causal inference in great detail.

## Features
--------

- **Support for Advanced Models**  
  Leverage state-of-the-art statistical models for causal impact analysis, including:  
  - **TensorFlow Probability**: Bayesian Structural Time Series (BSTS) models with support for trend, seasonality, and regression components.  
  - **Prophet**: Time series forecasting with robust handling of seasonality, missing data, and external regressors.  
  - **Pyro**: Bayesian regression using Variational Inference (VI) or Hamiltonian Monte Carlo (HMC) for probabilistic modeling and uncertainty quantification.  

- **Adapter-Based Modular Design**  
  Easily extend the library by integrating custom models. The adapter-based architecture allows seamless addition of new frameworks.  

- **Highly Configurable**  
  Fine-tune model parameters, specify covariates, and select fitting methods (e.g., HMC, VI) to tailor analyses to specific needs.  

- **Rigorous Evaluation**  
  Includes tools for pre- and post-intervention analysis, model performance assessment, and confidence interval computation for causal inference.  

- **Powerful Visualization**  
  Generate insightful visualizations, including forecasts, confidence intervals, and intervention effects, to better interpret and communicate results.  

## Use Cases & Examples
--------

CImpact is versatile and can be applied to various domains to measure the causal impact of interventions. Here are some examples:

- **Marketing Campaigns**: Assess the impact of a marketing campaign on sales over time using time series data.
- **Healthcare**: Evaluate the effect of a new drug or treatment on patient outcomes over a period.
- **Economic Policy**: Measure the impact of a new economic policy or regulatory change on key economic indicators.

Explore the `examples/` directory in this repository for further use case examples and code templates.

## Code Structure
------------

```plaintext
CImpact/
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

### 1. Stable Release

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
git clone https://github.com/Sanofi-Public/CImpact.git
cd CImpact
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
data = pd.read_csv('https://raw.githubusercontent.com/Sanofi-Public/CImpact/master/examples/google_data.csv')

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
ci = 95                            # Desired confidence interval

# Run the analysis
analysis = CausalImpactAnalysis(data, pre_period, post_period, model_config, index_col, target_col, observed_color,  predicted_color, ci_color, intervention_color, ci)
result = analysis.run_analysis()
print(result)
```


##### Outcome

![Result visualization for Tensorflow model](https://github.com/Sanofi-Public/CImpact/blob/master/examples/results/tensorflow_google_data_results.png "Result visualization for Tensorflow model")

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
> Please refer to [`examples/how-to-use.md`](./examples/how-to-use.md) for detailed model configuration instructions and additional usage examples of the library.

## Evaluation Methods
------------------

CImpact offers comprehensive tools to evaluate model performance and quantify the causal impact of interventions:  

- **Summary Statistics**  
  Obtain detailed point estimates, confidence intervals, and probabilistic measures of the intervention's impact.  

- **Impact Visualization**  
  Generate intuitive plots that display:  
  - Observed data versus counterfactual predictions.  
  - Estimated impact over time, including uncertainty intervals.  

- **Model Diagnostics**  
  Conduct residual analysis and access diagnostic metrics to evaluate model fit and robustness.  

## Performance Comparison
----------------------

CImpact supports a variety of models, each with unique strengths:  

- **TensorFlow**  
  Delivers robust performance with flexibility for advanced inference techniques, such as Variational Inference (VI) and Hamiltonian Monte Carlo (HMC).  

- **Prophet**  
  Offers a user-friendly experience with built-in support for seasonality and holiday effects. While effective for many use cases, it may exhibit slower performance on larger datasets.  

- **Pyro**  
  Excels in Bayesian inference, enabling powerful probabilistic modeling. However, its computational demands can be higher compared to other models.  

## Model Comparison & Best Practices
----------------------

Each model in CImpact has unique advantages, and selecting the right model can significantly impact your results. Here are some recommendations for selecting a model based on your use case:

- **Prophet**: Best for time series data with clear seasonal patterns and holidays, and if interpretability and ease of use are a priority. However, it might struggle with very large datasets or complex causal relationships.
  
- **TensorFlow (Bayesian Structural Time Series)**: Ideal for users who need advanced Bayesian modeling and have computational resources for methods like HMC and VI. It works well for more complex time series data with multiple covariates and non-linearities.

- **Pyro**: Choose this model if you need a fully probabilistic approach for causal inference. It's perfect for those who need flexibility with custom priors and Bayesian inference but are comfortable with Pyro's steeper learning curve.

Make sure to benchmark each model using your data before committing to one, and consider running a few trials with different models to compare their performance.

## Future Plans
------------

We welcome contributions to enhance and refine the library. While we are particularly interested in contributions in the following areas, we are open to other suggestions as well. If you have any ideas, please create an issue to discuss potential contributions.

- Add new, qualified models to broaden analytical options. We are currently exploring zero-shot learning models like [Google timesfm](https://github.com/google-research/timesfm) or [Amazon Chronos](https://www.amazon.science/code-and-datasets/chronos-learning-the-language-of-time-series).
- Enhanced Visualization**: Develop advanced plotting functions for deeper insights and a better understanding of results.
- Publish detailed tutorials to help users in effectively utilizing the library.

## Contributing
------------

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to participate.

## Contributors
-------
We would like to acknowledge the following individuals for their contributions to the development of this open-source library:
- **Amin Kamaleddin**
- **Diplumar Patel**
- **Charles Girard**
- **Nitesh Soni**

## License
-------

This work is available for academic research and non-commercial use only. See the <LICENSE> file for details.
CImpact is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it with attribution.

## Acknowledgements
----------------
We are thankful of Google research (cited below[^1]) team for publishing **"Inferring causal impact using Bayesian structural time-series models"** research paper and sharing orginal [R package](https://github.com/google/CausalImpact) to open souce community. We also extend our gratitude to the authors of [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) for their foundational work, which inspired this library.  


[^1]: Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015). Inferring causal impact using Bayesian structural time-series models.

* * * * *
