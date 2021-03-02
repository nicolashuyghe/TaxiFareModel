from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse


class Trainer():
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
        pipe_distance = Pipeline([
            ('dist_transformer', DistanceTransformer()),
            ('scaler', StandardScaler())
        ])

        # create time pipeline
        pipe_time = Pipeline([
            ('time_transformer', TimeFeaturesEncoder(time_column='pickup_datetime')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # create preprocessing pipeline
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        pipe_preproc = ColumnTransformer([
            ('distance',  pipe_distance, dist_cols),
            ('time', pipe_time, time_cols)],
            remainder='drop'
        )

        # Add the model of your choice to the pipeline
        self.pipeline = Pipeline([
            ('preprocessing', pipe_preproc),
            ('linear_regression', LinearRegression())])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        # compute y_pred on the test set
        y_pred = self.pipeline.predict(X_test)

        # call compute_rmse
        self.result = compute_rmse(y_pred, y_test)
        return self.result


if __name__ == "__main__":
    # get data
    df = get_data(nrows=10_000)

    # clean data
    df_cleaned = clean_data(df, test=False)

    # set X and y
    X = df_cleaned.drop(columns='fare_amount')
    y = df_cleaned['fare_amount']

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()

    # evaluate
    result = trainer.evaluate(X_test, y_test)
    print(result)
