import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(data_dir):
    X = []
    y = []
    labels = os.listdir(data_dir)
    
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(label_dir, file))
                    X.append(data)
                    y.append(label)
    
    return np.array(X), np.array(y)

def train_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def main():
    # Load training data
    X, y = load_data('data/train')
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the classifier
    clf = train_classifier(X_train, y_train)
    
    # Validate the classifier
    y_pred = clf.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    
    # Save the trained model
    joblib.dump(clf, 'gesture_classifier.pkl')

if __name__ == "__main__":
    main()