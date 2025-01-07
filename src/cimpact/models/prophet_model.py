""" 
Prophet model for causal impact measurement
"""

from prophet import Prophet
import numpy as np
from cimpact.models.base_model import BaseModel


class ProphetModel(BaseModel):
    """
    Modeling class for the Prophet model, extending the Base Model.
    This class provides methods to fit the model, make predictions, and evaluate model performance.
    """

    # pylint: disable=too-many-arguments
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
        self.covariates = covariates
        self.model = None

    def fit(self):
        """
        Fit the Prophet model using the pre-intervention data.
        """
        self.preprocess_data()
        df_train = self.data.reset_index().rename(
            columns={self.index_col: "ds", self.target_col: "y"}
        )
        self.model = Prophet()
        for cov in self.covariates.columns:
            self.model.add_regressor(cov)
        pre_train = df_train.iloc[: self.pre_period[1] + 1]
        self.model.fit(pre_train)

    def predict(self):
        """
        Make predictions using the Prophet model.

        Returns:
        - post_pred (np.array): Predictions for the post-intervention period.
        - pre_pred (np.array): Predictions for the pre-intervention period.
        - combined_predictions (np.array): Combined predictions for the full period.
        - forecast (pd.DataFrame): Full forecast DataFrame from Prophet.
        """
        future = self.model.make_future_dataframe(
            periods=len(self.post_data), include_history=True
        )
        for cov in self.covariates.columns:
            future[cov] = self.data[cov].values
        forecast = self.model.predict(future)

        pre_pred = forecast.iloc[self.pre_period[0]: self.pre_period[1] + 1]["yhat"]
        post_pred = forecast.iloc[
            self.pre_period[1] + 1 : self.pre_period[1] + 1 + len(self.post_data)
        ]["yhat"]
        combined_predictions = np.concatenate([pre_pred.values, post_pred.values])

        return post_pred.values, pre_pred.values, combined_predictions, forecast
