#!/usr/bin/env python3
"""
Demo script showcasing Emilio Ranucoli's Python/ML skills.
Features: data analysis, ML model training, and API integration.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests
import json


def fetch_sample_data():
    """Fetch sample data from a public API."""
    try:
        response = requests.get("https://jsonplaceholder.typicode.com/todos/1")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def train_simple_model():
    """Train a simple ML model using scikit-learn."""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.2f}")
    return model


def main():
    """Run the demo script."""
    print("=== Emilio Ranucoli Portfolio Demo ===")
    print("\n1. Fetching sample data...")
    data = fetch_sample_data()
    print(f"Sample data: {json.dumps(data, indent=2)}")

    print("\n2. Training a simple ML model...")
    model = train_simple_model()

    print("\n3. Demo completed successfully! 🎉")


if __name__ == "__main__":
    main()
