import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the data for training.
    
    Parameters:
    data (pd.DataFrame): Raw data.
    
    Returns:
    pd.DataFrame, pd.Series: Features and labels.
    """
    # Assuming the label column is named 'label' and the rest are features
    X = data.drop(columns=['label'])
    y = data['label']
    return X, y

def train_model(X, y):
    """
    Train a RandomForest model.
    
    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Labels.
    
    Returns:
    RandomForestClassifier: Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model

def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Parameters:
    model (RandomForestClassifier): Trained model.
    file_path (str): Path to save the model.
    """
    joblib.dump(model, file_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model to predict mind wandering.")
    parser.add_argument("-d", "--data", type=str, required=True, help="Path to the training data file (CSV)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the trained model")

    args = parser.parse_args()

    # Load and preprocess data
    data = load_data(args.data)
    X, y = preprocess_data(data)

    # Train model
    model = train_model(X, y)

    # Save model
    save_model(model, args.output)
