# Cimpact

CausalImpact is a modular causal impact analysis library with support for multiple time series models, including TensorFlow, Prophet, and Pyro. This library provides a flexible framework for estimating the causal effect of an intervention on time series data.

## Introduction

CausalImpact is designed to help analysts and data scientists assess the impact of an intervention on time series data. By leveraging different models, this library aims to provide robust causal inference results, accommodating various use cases and preferences in model selection.

## Why This Library?

This library extends the functionalities of the [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) library by incorporating support for multiple modeling approaches, including TensorFlow, Prophet, and Pyro. This modular design allows users to choose the best model for their specific needs and compare the performance and results across different models.

## Credits

This library is inspired by and builds upon the [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) library. We would like to acknowledge the contributions of the original authors and thank them for their work, which has been a foundation for this extended version.

## Improvements Over tfcausalimpact

- **Support for Multiple Models**: In addition to TensorFlow, we have added support for Prophet and Pyro models.
- **Modular Design**: Each model is implemented as a separate adapter, making it easy to extend the library with new models in the future.
- **Enhanced Flexibility**: Users can now choose from a variety of model configurations and hyperparameters to better suit their specific analysis needs.

## Advantages of Modular Components

- **Extensibility**: Adding new models or updating existing ones is straightforward.
- **Flexibility**: Users can compare different models' performance and results, choosing the best approach for their data.
- **Maintainability**: The modular design ensures that changes in one component do not affect others, making the library easier to maintain.

## Evaluation Methods

The library includes comprehensive evaluation methods to assess model performance and the causal impact of interventions. These methods are integrated into the framework, ensuring consistent and reliable results.

## How to Run the Code

Here is an example of how to use the CausalImpact library:

```python
import pandas as pd
from cimpact import CausalImpactAnalysis

# Define your data
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

# Define the pre and post periods
pre_period = ['2020-01-01', '2020-06-01']
post_period = ['2020-06-02', '2020-12-31']

# Define the frequency of data
freq = "D"

# Run the analysis
analysis = CausalImpactAnalysis(data, pre_period, post_period, model_config, 'DATE', 'TARGET', freq)
result = analysis.run_analysis()
print(result)
```

## Model Configurations

### TensorFlow Model
- standardize: Whether to standardize the data.
- learning_rate: Learning rate for the optimizer.
- num_variational_steps: Number of steps for variational inference.
- fit_method: Method for fitting the model (‘vi’ for variational inference, ‘hmc’ for Hamiltonian Monte Carlo).

### Prophet Model

- standardize: Whether to standardize the data.
- Additional parameters can be passed directly to the Prophet model.

### Pyro Model

- standardize: Whether to standardize the data.
- learning_rate: Learning rate for the optimizer.
- num_iterations: Number of iterations for training.
- num_samples: Number of samples for prediction.

## Performance Comparison

We have compared the performance of TensorFlow, Prophet, and Pyro models on various causal inference tasks. Here are some key observations:

- TensorFlow: Offers robust performance and flexibility with advanced features like variational inference and HMC.
- Prophet: Simple to use with built-in handling of seasonality and holidays, but may be slower for large datasets.
- Pyro: Provides powerful Bayesian inference capabilities, but requires more computational resources.

## Future Plans

- Expand Model Support: Add more models to the library to provide even more options for users.
- Enhanced Visualization: Improve the visualization capabilities for better insights and interpretation of results.
- User Contributions: Encourage community contributions to extend and improve the library.

## License

This work is shared under Apache Licence v 2

## Acknowledgements

We would like to thank the authors of tfcausalimpact for their foundational work, which inspired and enabled the creation of this extended library.