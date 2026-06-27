import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


BASE_DIR = Path(__file__).resolve().parent

def data_load_splitting(filename:str):

    dataset_path = BASE_DIR / "dataset"/filename

    df = pd.read_csv(dataset_path)

    x = df.drop(columns='academic_performance')
    y = df['academic_performance']

    x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)

    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test = data_load_splitting('Teen_Mental_Health_dataset.csv')
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

def pipeline():

    ohe_pipe = Pipeline(steps=[
        ('ohe',OneHotEncoder(drop='first',handle_unknown='ignore'))
    ])

    ord_encode = Pipeline(steps=[
        ('ord',OrdinalEncoder(categories=[['low', 'medium', 'high']]))
    ])

    num_pipe = Pipeline(steps=[
        ('scale',StandardScaler())
    ])
    
    process = ColumnTransformer(transformers=[
        ('ohe',ohe_pipe,make_column_selector(pattern=['gender','platform_usage']))
    ])

    model_pipe = Pipeline(steps=[
        ('process')
    ])

def model_train():
    pass
