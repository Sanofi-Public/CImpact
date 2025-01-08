""" 
Base model class which needs to be extended by all the models.
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cimpact.utils import convert_dates_to_indices


class BaseModel(ABC):
    """
    Abstract Base Model class to be extended by all specific model implementations.
    This class provides common functionality for fitting, predicting, evaluating,
    and handling time series data for causal impact analysis.
    """

    #pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(
        self, data, pre_period, post_period, index_col, target_col, model_args=None
    ):
        """ 
        Constructor method of base model class
        """
        self.data = self.validate_and_format_data(data)
        self.pre_period = pre_period
        self.post_period = post_period
        self.model_args = model_args
        self.model = None
        self.inferences = None
        self.target_col = target_col
        self.index_col = index_col
        self.post_data = None
        self.mean = None
        self.std = None
        self.pre_data = None

    @abstractmethod
    def fit(self):
        """
        Abstract method for fitting the model to data.
        This method must be implemented by subclasses.
        """
        return

    @abstractmethod
    def predict(self):
        """
        Abstract method for making predictions using the fitted model.
        This method must be implemented by subclasses.
        """
        return

    def evaluate(self):
        """
        Evaluate the model performance using Root Mean Squared Error (RMSE).

        Returns:
        - float: RMSE value.
        """
        predictions, _, _, _ = self.predict() #pylint: disable=too-many-locals
        return np.sqrt(np.mean((self.data[self.target_col].values - predictions) ** 2))


    def plot(
        self,
        combined_predictions,
        observed_color,           # Sanofi's black
        predicted_color,          # Sanofi's purple
        ci_color,                 # Light purple with transparency for confidence interval
        intervention_color,       # Dark gray for intervention line
        figsize,                  # Desired figure size
        zscore                    # Zscore to plot confidence interval
    ):
        """
        Function to plot the results with Sanofi brand color options.

        Recommended Color Palettes:

        1. **Sanofi Brand Colors** (Purple and Black)
        observed_color: "#000000" (Black)
        predicted_color: "#7A00E6" (Sanofi purple)
        ci_color: "#D9B3FF66" (Light lavender with transparency)
        intervention_color: "#444444" (Dark gray)

        2. **Purple Theme**
        observed_color: "#6A0DAD" (Dark purple)
        predicted_color: "#9C27B0" (Medium purple)
        ci_color: "#E6B3E666" (Lavender with transparency)
        intervention_color: "#8A2BE2" (Blue-purple)

        3. **Orange Theme**
        observed_color: "#FF4500" (Orange-red)
        predicted_color: "#FF8C00" (Dark orange)
        ci_color: "#FFDAB966" (Peach with transparency)
        intervention_color: "#D2691E" (Chocolate)

        4. **Blue Theme**
        observed_color: "#1E90FF" (Dodger blue)
        predicted_color: "#4682B4" (Steel blue)
        ci_color: "#B0C4DE66" (Light steel blue with transparency)
        intervention_color: "#4169E1" (Royal blue)

        5. **Green Theme**
        observed_color: "#228B22" (Forest green)
        predicted_color: "#32CD32" (Lime green)
        ci_color: "#98FB9866" (Pale green with transparency)
        intervention_color: "#006400" (Dark green)
        """
        if self.inferences is None:
            raise ValueError("Model must be fitted before plotting.")

        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

        full_data = self.data[self.pre_period[0]:][self.target_col]
        predicted_means = (
            combined_predictions
            if combined_predictions is not None
            else self.inferences["predicted_mean"]
        )
        ci_lower_full = predicted_means - zscore * np.std(predicted_means) 
        ci_upper_full = predicted_means + zscore * np.std(predicted_means) 
        
        ci_lower_effect = (full_data - predicted_means ) - zscore * np.std(full_data - predicted_means ) #  predicted - full_data
        ci_upper_effect = (full_data - predicted_means ) + zscore * np.std(full_data - predicted_means ) #  predicted - full_data

        for i, panel in enumerate(["original", "pointwise", "cumulative"]):
            ax = axs[i]
            if panel == "original":
                ax.plot(full_data.index, full_data, color=observed_color, label="Observed")
                ax.plot(full_data.index, predicted_means, linestyle="--", color=predicted_color, label="Predicted")
                ax.fill_between(full_data.index, ci_lower_full, ci_upper_full, color=ci_color)
            
            elif panel == "pointwise":
                ax.plot(full_data.index, full_data-predicted_means, linestyle="--", color=predicted_color, label="Point effects") # fix predicted_means-full_data iso predicted_means
                ax.fill_between(full_data.index, ci_lower_effect, ci_upper_effect, color=ci_color)
                ax.axhline(y=0, color="xkcd:light grey", linestyle="--")

            
            elif panel == "cumulative":
                point_effects_post = self.post_data[self.target_col].values - predicted_means[-len(self.post_data):]
                cumulative_effects = np.cumsum(point_effects_post)
                ax.plot(self.post_data.index, cumulative_effects, linestyle="--", color=predicted_color, label="Cumulative Effects")
                ax.fill_between(
                    self.post_data.index,
                    cumulative_effects - zscore * np.std(cumulative_effects),
                    cumulative_effects + zscore * np.std(cumulative_effects), 
                    color=ci_color,
                )
                ax.axhline(y=0, color="xkcd:light grey", linestyle="--")

            ax.axvline(self.data.index[self.pre_period[1]], color=intervention_color, linestyle="--", label="Intervention")
            ax.legend()

        plt.tight_layout()
        return fig

    def preprocess_data(self):
        """
        Preprocess the data by validating, formatting, and segmenting it into pre and post periods.
        
        Returns:
        - tuple: The original data, pre-intervention data, and post-intervention data.
        """
        self.data = self.validate_and_format_data(self.data)
        self.pre_data, self.post_data = self.segment_data(
            self.data, self.pre_period, self.post_period
        )
        return self.data, self.pre_data, self.post_data

    def postprocess_results(self, forecast, pre_pred, combined_predictions, zscore): #pylint: disable=unused-argument
        """
        Postprocess the model's results to generate inferences.
        """
        # Ensure forecast is a full series of predictions
        if len(forecast.shape) == 1 or forecast.shape[1] == 1:
            forecast = forecast.ravel()  # Flatten array if it's 2D but only 1 column

        # Ensure that the indices are numeric
        if isinstance(self.post_period[0], str):
            self.post_period = convert_dates_to_indices(self.data, self.post_period)

        # Extracting the post-period data from self.data['y']
        # post_data = self.data[self.target_col].values

        self.inferences = {
            "predicted_mean": forecast,
            "ci_lower": forecast - zscore * np.std(forecast),  # Example calculation
            "ci_upper": forecast + zscore * np.std(forecast),
            "point_effects": forecast
            - self.post_data[self.target_col].values,  # Adjusted to match lengths
            "cumulative_effects": np.cumsum(
                forecast - self.post_data[self.target_col].values
            ),  # Adjusted to match lengths
        }

    def validate_and_format_data(self, data):
        """
        Method to validate and format the input data.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if not np.all(np.isfinite(data.select_dtypes(include=[np.number]))):
            raise ValueError("Data contains non-finite values.")
        return data

    def segment_data(self, data, pre_period, post_period):
        """
        Segments the data into pre and post periods using integer indices.

        Args:
            data (pd.DataFrame): The DataFrame from which to extract segments.
            pre_period (list): A list of two integers [start_index, end_index] for the pre period.
            post_period (list): A list of two integers [start_index, end_index] for the post period.

        Returns:
            tuple: A tuple containing two DataFrames (pre_data, post_data).
        """
        # Use pandas integer-location based indexing
        pre_data = data.iloc[
            pre_period[0] : pre_period[1] + 1
        ]  # +1 because pandas slicing is exclusive on the end index
        post_data = data.iloc[
            post_period[0] : post_period[1] + 1
        ]  # Similarly, add 1 to include the end index

        return pre_data, post_data

    def standardize_data(self, data):
        """ 
        Method to standardize the data using the mean and standard deviation of the target column.
        """
        numeric_cols = data.select_dtypes(include=["number"]).columns
        self.mean = data[numeric_cols].mean()
        self.std = data[numeric_cols].std()

        # Standardize numerical columns
        data[numeric_cols] = (data[numeric_cols] - self.mean) / self.std

        return data, (self.mean, self.std)

    def destandardize_data(self, data):
        """ 
        Method to destandardize the data using the mean and standard deviation of the target column.
        """
        # return data * self.std[self.target_col] + self.mean[self.target_col] ## TODO: Changed when writing unit test cases
        numeric_data = data.select_dtypes(include=[np.number])
        return numeric_data * self.std[numeric_data.columns] + self.mean[numeric_data.columns]