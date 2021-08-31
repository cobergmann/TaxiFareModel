import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property


class Trainer(BaseEstimator, TransformerMixin):

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                        remainder="drop")
        pipeline = Pipeline([('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())])
        self.mlflow_log_param('Model', 'LinearRegression')
        return pipeline

    def run(self):
        """set and train the pipeline"""

        # set pipeline and fit it
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

        # mlflow helper
        experiment_id = self.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

        # return fitted pipeline
        return self.pipeline

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        return joblib.dump(self.pipeline, 'model.joblib')

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        # print(rmse)
        self.mlflow_log_metric('RMSE', rmse)
        return rmse


    ## MLFLOW

    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[PT] [Lisboa] [cberg] TaxiFareModel + 1"

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    data = get_data()
    # clean data
    data = clean_data(data)
    # set X and y
    y = data.fare_amount
    X = data.drop(columns='fare_amount')
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3)
    # train
    trainer = Trainer(X_train, y_train)
    pipeline = trainer.set_pipeline()
    trainer.run()
    # save model
    trainer.save_model()
    # evaluate
    trainer.evaluate(X_test, y_test)
