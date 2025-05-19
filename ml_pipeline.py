import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    # Drop irrelevant columns
    df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])
    
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df[['Age']])
    
    # Fill Embarked with most frequent
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Convert categorical variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Scale numeric columns
    scaler = StandardScaler()
    numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def train_model(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def main():
    df = load_data()
    df_processed = preprocess_data(df)
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc*100:.2f}%")
    
    # Save model and preprocessing objects
    joblib.dump(model, "random_forest_model.joblib")

if __name__ == "__main__":
    main()
