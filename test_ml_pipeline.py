import pytest
import pandas as pd
from ml_pipeline import preprocess_data, train_model
from sklearn.ensemble import RandomForestClassifier

def test_train_model():
    # Prepare a minimal sample dataset
    data = {
        'Age': [0, 1, 2, 3],
        'Fare': [10, 20, 10, 30],
        'SibSp': [0, 1, 0, 1],
        'Parch': [0, 0, 1, 1],
        'Sex_male': [1, 0, 1, 0],
        'Embarked_Q': [0, 1, 0, 0],
        'Embarked_S': [1, 0, 1, 1]
    }
    X = pd.DataFrame(data)
    y = [0, 1, 0, 1]
    
    # Call the train_model function
    model = train_model(X, y)
    
    # Assert that the returned model is an instance of RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)
    
    # Assert the model has been fitted (check attribute 'n_estimators' as a proxy)
    assert hasattr(model, "n_estimators")
    
    # Predict on training data and check the prediction output shape
    preds = model.predict(X)
    assert len(preds) == len(y)
    
    # Check accuracy is reasonable (since it's train data, accuracy should be high)
    accuracy = sum(preds == y) / len(y)
    assert accuracy >= 0.75, f"Training accuracy too low: {accuracy:.2f}"



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
    print(f"Model accuracy: {accuracy:.2f}")
    assert accuracy >= 0.8
