import pytest
import pandas as pd
from ml_pipeline import preprocess_data, train_model

def test_preprocess_data():
    sample_data = {
        'Age': [22, None, 38],
        'Fare': [7.25, 71.83, 8.05],
        'SibSp': [1, 1, 0],
        'Parch': [0, 0, 0],
        'Sex': ['male', 'female', 'female'],
        'Embarked': ['S', 'C', None],
        'Survived': [0, 1, 1],
        'Name': ['A', 'B', 'C'],
        'Ticket': ['123', '456', '789'],
        'Cabin': [None, 'C85', None],
        'PassengerId': [1, 2, 3]
    }
    df = pd.DataFrame(sample_data)
    df_processed = preprocess_data(df)
    
    # Check no missing values
    assert df_processed.isnull().sum().sum() == 0
    # Check categorical columns replaced
    assert 'Sex_male' in df_processed.columns

def test_model_accuracy():
    # minimal dataset to check if model trains
    data = {
        'Age': [0, 1],
        'Fare': [0, 1],
        'SibSp': [0, 0],
        'Parch': [0, 0],
        'Sex_male': [1, 0],
        'Embarked_Q': [0, 1],
        'Embarked_S': [1, 0]
    }
    X = pd.DataFrame(data)
    y = [0, 1]
    
    model = train_model(X, y)
    preds = model.predict(X)
    accuracy = sum(preds == y) / len(y)
    assert accuracy >= 0.8
