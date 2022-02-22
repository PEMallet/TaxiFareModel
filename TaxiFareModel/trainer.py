## Imports locaux
from  .data import get_data
from .data import clean_data
from .encoders import DistanceTransformer
from .encoders import TimeFeaturesEncoder
from .utils import compute_rmse
## Imports python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

  # implement set_pipeline() function
    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self


    def run(self):
    # implement train() function
        '''returns a trained pipelined model'''
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    print ('>> Open Data')
    df = get_data(100)

    # clean data
    print ('>> Clean Data')
    df = clean_data(df)

    # set X and y
    y = df.pop("fare_amount")
    X = df

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train
    print ('>> Implementation et fit du modele')
    train =  Trainer(X_train, y_train)
    train.set_pipeline().run()

    # evaluate
    print ('>> Evaluate (RMSE)')
    print ( train.evaluate(X_test, y_test)  )
