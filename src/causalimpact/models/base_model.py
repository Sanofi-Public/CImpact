from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from causalimpact.utils import convert_dates_to_indices

class BaseModel(ABC):
    def __init__(self, data, pre_period, post_period, index_col, target_col, model_args=None):
        self.data = self.validate_and_format_data(data)
        self.pre_period = pre_period
        self.post_period = post_period
        self.model_args = model_args
        self.model = None
        self.inferences = None
        self.target_col = target_col
        self.index_col = index_col

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def evaluate(self):
        """
        Evaluate the model performance using Root Mean Squared Error (RMSE).

        Returns:
        - float: RMSE value.
        """
        predictions, _, _, _ = self.predict()
        return np.sqrt(np.mean((self.data[self.target_col].values - predictions) ** 2))

    def plot(self, combined_predictions=None):
        if self.inferences is None:
            raise ValueError("Model must be fitted before plotting.")
        
        figsize = (10, 7)
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        full_data = self.data[self.target_col]
        predicted_means = combined_predictions if combined_predictions is not None else self.inferences['predicted_mean']
        ci_lower_full = combined_predictions - 1.96 * np.std(combined_predictions)
        ci_upper_full = combined_predictions + 1.96 * np.std(combined_predictions)
        
        # Improved visual settings
        observed_color = 'black'
        predicted_color = 'orangered'
        ci_color = (1.0, 0.4981, 0.0549, 0.4)  # RGBA with alpha
        
        # Dynamic subplot adjustment and handling
        for i, panel in enumerate(['original', 'pointwise', 'cumulative']):
            ax = axs[i]
            if panel == 'original':
                ax.plot(full_data.index, full_data, color=observed_color, label='Observed')
                ax.plot(full_data.index, predicted_means, linestyle='--', color=predicted_color, label='Predicted')  # Skip the first point
                ax.fill_between(full_data.index, ci_lower_full, ci_upper_full, color=ci_color, alpha=0.4)
            elif panel == 'pointwise':
                # point_effects = full_data - predicted_means
                ax.plot(full_data.index, predicted_means, linestyle='--', color=predicted_color, label='Point effects')
                ax.fill_between(full_data.index, ci_lower_full, ci_upper_full, color=ci_color, alpha=0.4)
            elif panel == 'cumulative':
                point_effects_post = self.post_data[self.target_col].values - predicted_means[-len(self.post_data):]
                cumulative_effects = np.cumsum(point_effects_post)
                ax.plot(self.post_data.index, cumulative_effects, linestyle='--', color=predicted_color, label='Cumulative Effects')
                ax.fill_between(self.post_data.index, 
                                cumulative_effects - 1.96 * np.std(cumulative_effects), 
                                cumulative_effects + 1.96 * np.std(cumulative_effects), 
                                color=ci_color, alpha=0.4)
                ax.axhline(y=0, color='gray', linestyle='--')
            
            ax.axvline(self.data.index[self.pre_period[1]], color='gray', linestyle='--', label='Intervention')
            ax.legend()
        
        plt.tight_layout()
        # plt.show()
        # plt.close(fig)  # Close the plot to prevent it from displaying automatically
        return fig

    def preprocess_data(self):
        self.data = self.validate_and_format_data(self.data)
        self.pre_data, self.post_data = self.segment_data(self.data, self.pre_period, self.post_period)
        return self.data, self.pre_data, self.post_data

    def postprocess_results(self, forecast, pre_pred, combined_predictions):
        # Ensure forecast is a full series of predictions
        if len(forecast.shape) == 1 or forecast.shape[1] == 1:
            forecast = forecast.ravel()  # Flatten array if it's 2D but only 1 column
        
        # Ensure that the indices are numeric
        if isinstance(self.post_period[0], str):
            self.post_period = convert_dates_to_indices(self.data, self.post_period)
        
        # Extracting the post-period data from self.data['y']
        post_data = self.data[self.target_col].values
        
        self.inferences = {
            'predicted_mean': forecast,
            'ci_lower': forecast - 1.96 * np.std(forecast),  # Example calculation
            'ci_upper': forecast + 1.96 * np.std(forecast),
            'point_effects': forecast - self.post_data[self.target_col].values,  # Adjusted to match lengths
            'cumulative_effects': np.cumsum(forecast - self.post_data[self.target_col].values)  # Adjusted to match lengths
        }

    def validate_and_format_data(self, data):
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
        pre_data = data.iloc[pre_period[0]:pre_period[1]+1]  # +1 because pandas slicing is exclusive on the end index
        post_data = data.iloc[post_period[0]:post_period[1]+1]  # Similarly, add 1 to include the end index

        return pre_data, post_data
    
    def standardize_data(self, data):
        numeric_cols = data.select_dtypes(include=['number']).columns
        self.mean = data[numeric_cols].mean()
        self.std = data[numeric_cols].std()

        # Standardize numerical columns
        data[numeric_cols] = (data[numeric_cols] - self.mean) / self.std

        return data, (self.mean, self.std)   
    
    def destandardize_data(self, data):
        return data * self.std[self.target_col] + self.mean[self.target_col]