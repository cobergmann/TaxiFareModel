import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data


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
        return pipeline

    def run(self):
        """set and train the pipeline"""

        # set pipeline and fit it
        self.pipeline = Trainer.set_pipeline(self)
        self.pipeline.fit(self.X, self.y)
        return self.pipeline


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test)**2).mean())
        # print(rmse)
        return rmse


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
    # evaluate
    trainer.evaluate(X_test, y_test)
