import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

class MatchPredictorModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.classifier = None
        self.algorithm = 'naive_bayes'  # default
        self.label_encoders = {}
        self.label_encoder_result = None
        self.categorical_columns = ['Day/Night', 'Opponent', 'Batting_First/Second', 
                                     'Recent_Form', 'Competition_Type', 'Venue']
        self.feature_names = [] 
        self.accuracy = 0.0
        self.confusion_mat = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        # Train model on initialization
        self.train_model()
        
    def set_algorithm(self, algorithm):
        """Set the algorithm to use: 'naive_bayes' or 'id3'"""
        self.algorithm = algorithm
        
    def train_model(self):
        matches_data = pd.read_csv(self.csv_path)
        self.df = matches_data.copy()
        matches_data = matches_data.drop('Date', axis=1)
        
        for column in self.categorical_columns:
            le = LabelEncoder()
            matches_data[column] = le.fit_transform(matches_data[column])
            self.label_encoders[column] = le
        
        self.label_encoder_result = LabelEncoder()
        matches_data['Result'] = self.label_encoder_result.fit_transform(matches_data['Result'])
        
        # Save a copy of the manipulated dataset for later display
        self.df_processed = matches_data.copy()
        
        
        X = matches_data.drop('Result', axis=1)
        self.feature_names = X.columns.tolist()
        Y = matches_data['Result']
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)
        
        # Select algorithm
        if self.algorithm == 'naive_bayes':
            self.classifier = GaussianNB()
        else:  # id3 (Decision Tree with entropy criterion)
            self.classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
        
        self.classifier.fit(self.X_train, self.Y_train)
        Y_predict = self.classifier.predict(self.X_test)
        
        self.accuracy = accuracy_score(self.Y_test, Y_predict)
        self.confusion_mat = confusion_matrix(self.Y_test, Y_predict)
        
    def get_confusion_matrix(self):
        """Return confusion matrix and class labels"""
        return self.confusion_mat, self.label_encoder_result.classes_
    
    def get_classifier(self):
        """Return the trained classifier for visualization"""
        return self.classifier
        
    def predict(self, input_dict):
        new_match = pd.DataFrame([input_dict])
        
        for col in self.categorical_columns:
            if col in new_match.columns:
                new_match[col] = self.label_encoders[col].transform(new_match[col])
        
        new_match = new_match[self.feature_names]
        prediction = self.classifier.predict(new_match)
        predicted_result = self.label_encoder_result.inverse_transform(prediction)[0]
        
        probabilities = self.classifier.predict_proba(new_match)[0]
        prob_map = dict(zip(self.label_encoder_result.classes_, probabilities))
        
        return predicted_result, prob_map