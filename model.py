import pandas as pd
from pathlib import Path
from colorama import Fore
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
filepath = 'Teen_Mental_Health_dataset.csv'

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


def load_data(filename:str):
    """
    Loads a CSV file from the 'dataset/' directory and returns it as a DataFrame.

    Args:
        filename (str): Name of the CSV file located inside the 'dataset/' folder.

    Returns:
        pd.DataFrame: The entire dataset as a pandas DataFrame.
    """
    
    dataset_path = BASE_DIR / "dataset"/filename

    return pd.read_csv(dataset_path)

def data_load_splitting(filename:str):
    """
    Loads the dataset, separates features from the target column,
    and splits the data into training and testing sets (80-20 ratio).

    Args:
        filename (str): Name of the CSV file to load.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
            - x_train: Training features
            - x_test:  Testing features
            - y_train: Training target (academic_performance)
            - y_test:  Testing target (academic_performance)
    """

    df = load_data(filename)

    x = df.drop(columns='academic_performance')
    y = df['academic_performance']

    x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)

    return x_train,x_test,y_train,y_test


def model_pipeline():
    """
    Builds and returns a complete sklearn Pipeline with preprocessing and a regression model.

    Preprocessing steps:
        - OneHotEncoder : Applied to 'gender' and 'platform_usage' columns.
        - OrdinalEncoder: Applied to 'social_interaction_level' (order: low < medium < high).
        - StandardScaler: Applied to all numeric (int/float) columns.

    Model:
        - SGDRegressor with ElasticNet regularization (max_iter=500).

    Returns:
        sklearn.pipeline.Pipeline: A fully configured preprocessing and model pipeline.
    """

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
        ('ohe',ohe_pipe,make_column_selector(pattern='gender|platform_usage')),

        ('ord',ord_encode,make_column_selector(pattern='social_interaction_level')),

        ('num_scale',num_pipe,make_column_selector(dtype_include=['int','float']))

    ],remainder='drop')

    model_pipe = Pipeline(steps=[
        ('process',process),
        ('model',SGDRegressor(penalty='elasticnet',
                                    max_iter=500))
    ])

    return model_pipe



def model_train(filename):
    """
    Trains the model pipeline, saves it to disk, and prints evaluation metrics on the test set.

    Steps:
        1. Builds the pipeline using model_pipeline().
        2. Loads and splits the data using data_load_splitting().
        3. Fits the pipeline on the training data.
        4. Generates predictions on the test data.
        5. Saves the trained pipeline to 'model/sgd_regressor.joblib' using joblib.
        6. Prints MAE, MSE, and R2 Score to the console.

    Args:
        filename (str): Name of the CSV dataset file.

    Returns:
        None: The trained model is saved to disk and evaluation results are printed to the console.
    """

    pipeline = model_pipeline()
    x_train,x_test,y_train,y_test = data_load_splitting(filename)

    pipeline.fit(x_train,y_train)
    y_pred = pipeline.predict(x_test)

    print(f'{Fore.LIGHTMAGENTA_EX} model is saved successfully')
    joblib.dump(pipeline, MODEL_DIR / "sgd_regressor.joblib")

    print(f'{Fore.CYAN} mean_absolute_error is {mean_absolute_error(y_test,y_pred)}')
    print(f'{Fore.CYAN} mean_squared_error is {mean_squared_error(y_test,y_pred)}')
    print(f'{Fore.GREEN} r2 score is {r2_score(y_test,y_pred)}')
    
if __name__== '__main__':
    model_train(filepath)