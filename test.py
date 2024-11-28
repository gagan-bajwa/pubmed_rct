# test.py
import tensorflow as tf
import numpy as np
from predict import predict_and_organize

# Test data (you can change this to any abstract text)
test_abstract = 'This RCT examined the efficacy of a manualized social intervention for children with HFASDs. Participants were randomly assigned to treatment or wait-list conditions. Treatment included instruction and therapeutic activities targeting social skills, face-emotion recognition, interest expansion, and interpretation of non-literal language. A response-cost program was applied to reduce problem behaviors and foster skills acquisition. Significant treatment effects were found for five of seven primary outcome measures (parent ratings and direct child measures). Secondary measures based on staff ratings (treatment group only) corroborated gains reported by parents. High levels of parent, child and staff satisfaction were reported, along with high levels of treatment fidelity. Standardized effect size estimates were primarily in the medium and large ranges and favored the treatment group.'


# Load the model
def load_model():
    # Make sure to load the model from the correct path
    model = tf.keras.models.load_model('model_1/full_model')
    return model

# Run prediction
def run_prediction():
    # Load the model
    model = load_model()
    
    # Get predictions from the model for the given abstract
    predictions = predict_and_organize(test_abstract, model)
    
    # Print the predictions
    print("Predictions:")
    for heading, sentences in predictions.items():
        print(f"\n{heading}:")
        for sentence in sentences:
            print(f"  - {sentence}")

if __name__ == "__main__":
    run_prediction()
