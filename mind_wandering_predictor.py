import pandas as pd
import joblib

class MindWanderingPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        """
        Predicts whether the subject is mind wandering based on input data.
        
        Parameters:
        input_data (pd.DataFrame): DataFrame containing the input features for prediction.
        
        Returns:
        pd.Series: Series containing the predictions.
        """
        predictions = self.model.predict(input_data)
        return pd.Series(predictions, name='mind_wandering')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict mind wandering from input data.")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input data file (CSV)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the predictions (CSV)")

    args = parser.parse_args()

    # Load input data
    input_data = pd.read_csv(args.input)

    # Initialize predictor
    predictor = MindWanderingPredictor(args.model)

    # Make predictions
    predictions = predictor.predict(input_data)

    # Save predictions to CSV
    predictions.to_csv(args.output, index=False)
