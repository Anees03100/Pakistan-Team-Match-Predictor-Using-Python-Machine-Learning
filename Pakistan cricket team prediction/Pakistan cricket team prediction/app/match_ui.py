from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QTextEdit, QSpinBox, 
                             QGroupBox, QGridLayout, QRadioButton, QButtonGroup, QDialog, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.tree import plot_tree, export_text
import seaborn as sns

class MatchPredictorApp(QMainWindow):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor 
        self.setWindowTitle("Pakistan Cricket Team Match Predictor")
        self.setGeometry(100, 100, 1000, 700)
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        title = QLabel()
        title.setText('<span style="font-size:18pt; font-weight:bold; color:#008000;">Pakistan Cricket Team Match Predictor</span>')
        title.setAlignment(Qt.AlignCenter)
        title.setTextFormat(Qt.RichText)
        main_layout.addWidget(title)
        
        algo_group = QGroupBox("Select Algorithm")
        algo_layout = QHBoxLayout()
        
        self.algo_button_group = QButtonGroup()
        self.nb_radio = QRadioButton("Naive Bayes")
        self.id3_radio = QRadioButton("ID3 (Decision Tree)")
        self.nb_radio.setChecked(True)
        
        self.algo_button_group.addButton(self.nb_radio)
        self.algo_button_group.addButton(self.id3_radio)
        
        algo_layout.addWidget(self.nb_radio)
        algo_layout.addWidget(self.id3_radio)
        
        self.train_button = QPushButton("Train Model")
        self.train_button.setStyleSheet("background-color: green; color: white; padding: 8px; border-radius: 5px;")
        self.train_button.clicked.connect(self.train_model)
        algo_layout.addWidget(self.train_button)
        
        algo_group.setLayout(algo_layout)
        main_layout.addWidget(algo_group)
        
        model_info_layout = QHBoxLayout()
        
        accuracy_pct = self.predictor.accuracy * 100
        self.accuracy_label = QLabel(f"Model Accuracy: {accuracy_pct:.2f}%")
        self.accuracy_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.accuracy_label.setStyleSheet("color: #008000;")
        model_info_layout.addWidget(self.accuracy_label)
        
        self.visualize_button = QPushButton("View Decision Rules")
        self.visualize_button.setStyleSheet("background-color: #4169E1; color: white; padding: 8px; border-radius: 5px;")
        self.visualize_button.clicked.connect(self.visualize_tree)
        self.visualize_button.setVisible(False)
        model_info_layout.addWidget(self.visualize_button)
        
        main_layout.addLayout(model_info_layout)
        
        self.confusion_label = QLabel("Confusion Matrix: Train a model first")
        self.confusion_label.setFont(QFont('Courier', 9))
        self.confusion_label.setStyleSheet("border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;")
        main_layout.addWidget(self.confusion_label)
        
        input_group = QGroupBox("Match Input Features")
        input_layout = QGridLayout()
        
        self.combos = {}
        self.score_spin = None
        
        row = 0
        for col in self.predictor.categorical_columns:
            input_layout.addWidget(QLabel(f"{col.replace('_', ' ').replace('/', ' or ').title()}:"), row, 0)
            combo = QComboBox()
            
            if col == 'Opponent':
                items = self.predictor.label_encoders[col].classes_
                for item in items:
                    display_text = f"{item}"
                    combo.addItem(display_text, item) 
            else:
                combo.addItems(self.predictor.label_encoders[col].classes_)
            
            input_layout.addWidget(combo, row, 1)
            self.combos[col] = combo
            row += 1
        
        input_layout.addWidget(QLabel("Score:"), row, 0)
        self.score_spin = QSpinBox()
        self.score_spin.setRange(50, 250)
        self.score_spin.setValue(150)
        input_layout.addWidget(self.score_spin, row, 1)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        self.predict_button = QPushButton("Predict Match Result")
        self.predict_button.setFont(QFont('Arial', 16, QFont.Bold))
        self.predict_button.setStyleSheet("background-color: #008000; color: white; padding: 10px; border-radius: 5px;")
        self.predict_button.clicked.connect(self.handle_prediction)
        main_layout.addWidget(self.predict_button)
        
        # Results Display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont('Courier', 10))
        main_layout.addWidget(self.results_text)
        

        algo_name = self.predictor.algorithm.replace('_', ' ').title()
        self.algorithm_label = QLabel(f"Current Algorithm: {algo_name}")
        self.algorithm_label.setFont(QFont('Arial', 9, QFont.Bold))
        self.algorithm_label.setStyleSheet("color: #484848;")
        model_info_layout.addWidget(self.algorithm_label)


        self.update_model_display()


    #yeh func algo choose krny k liye    
    def train_model(self):
        if self.nb_radio.isChecked():
            self.predictor.set_algorithm('naive_bayes')
            self.visualize_button.setVisible(False)
        else:
            self.predictor.set_algorithm('id3') 
            self.visualize_button.setVisible(True)
        
        self.predictor.train_model()
        self.update_model_display()
        
    def update_model_display(self):

        accuracy_pct = self.predictor.accuracy * 100
        self.accuracy_label.setText(f"Model Accuracy: {accuracy_pct:.2f}%")
    
        algo_name = self.predictor.algorithm.replace('_', ' ').title()
        self.algorithm_label.setText(f"Current Algorithm: {algo_name}")

        conf_mat, classes = self.predictor.get_confusion_matrix()
        conf_text = "Confusion Matrix:\n\n"
        conf_text += "Predicted →\n"
        conf_text += "Actual ↓     " + "  ".join([f"{cls:>8s}" for cls in classes]) + "\n"
    
        for i, cls in enumerate(classes):
            conf_text += f"{cls:>12s} "
            for j in range(len(classes)):
                conf_text += f"{conf_mat[i][j]:>8d}  "
            conf_text += "\n"

        # data set ki info
        df = self.predictor.df
        dataset_size = len(df)
        train_size = len(self.predictor.X_train)
        test_size = len(self.predictor.X_test)

        conf_text += "\n" + "-"*50 + "\n"
        conf_text += "DATASET INFORMATION\n"
        conf_text += "-"*50 + "\n"
        conf_text += f"Total Records     : {dataset_size}\n"
        conf_text += f"Training Set (80%): {train_size}\n"
        conf_text += f"Test Set (20%)    : {test_size}\n\n"

         #data set ki values k mean,std wagera dikhane k liye
        conf_text += "Dataset Statistical Summary (After Encoding):\n"
        conf_text += "-"*50 + "\n"
        conf_text += str(self.predictor.df_processed.describe(include='all')) + "\n"

        # Set text in the confusion matrix label
        self.confusion_label.setText(conf_text)

        
    def visualize_tree(self):
        """Display decision tree rules in a readable text format"""
        if self.predictor.algorithm != 'id3':
            self.results_text.setText("Visualization is only available for ID3 (Decision Tree) algorithm!")
            return
        
        classifier = self.predictor.get_classifier()
        _, classes = self.predictor.get_confusion_matrix()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Decision Tree Rules")
        dialog.setGeometry(200, 100, 900, 700)
        
        layout = QVBoxLayout()
        
        title = QLabel("📊 Decision Tree Rules (ID3 Algorithm)")
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title.setStyleSheet("color: #008000; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        tree_rules = export_text(classifier, 
                                 feature_names=self.predictor.feature_names,
                                 show_weights=True)
        
        text_display = QTextEdit()
        text_display.setReadOnly(True)
        text_display.setFont(QFont('Courier', 10))
        text_display.setStyleSheet("background-color: #f5f5f5; border: 2px solid #008000;")
        
        formatted_rules = self.format_tree_rules(tree_rules, classes)
        text_display.setText(formatted_rules)
        
        layout.addWidget(text_display)
        
        explanation = QLabel("💡 How to read: Follow the conditions from top to bottom. "
                           "Each 'class' shows the predicted match result.")
        explanation.setWordWrap(True)
        explanation.setStyleSheet("background-color: #ffffcc; padding: 8px; border: 1px solid #ccccaa;")
        layout.addWidget(explanation)
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("background-color: #666; color: white; padding: 8px;")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def format_tree_rules(self, rules_text, classes):
        lines = rules_text.split('\n')
        formatted = "DECISION TREE RULES\n"
        formatted += "=" * 70 + "\n\n"
        
        feature_map = {
            'Day/Night': 'Day/Night',
            'Opponent': 'Opponent Team',
            'Batting_First/Second': 'Batting Order',
            'Recent_Form': 'Recent Form',
            'Competition_Type': 'Competition Type',
            'Venue': 'Venue',
            'Score': 'Score'
        }
        
        for line in lines:
            if not line.strip():
                continue
            
            formatted_line = line
            for old, new in feature_map.items():
                formatted_line = formatted_line.replace(old, new)
            
            if 'class:' in formatted_line:
                for cls in classes:
                    if cls in formatted_line:
                        formatted_line = formatted_line.replace(f'class: {cls}', 
                                                               f'➤ PREDICT: {cls.upper()}')
            
            formatted += formatted_line + "\n"
        
        formatted += "\n" + "=" * 70 + "\n"
        formatted += f"\nTotal Decision Paths: {rules_text.count('class:')}\n"
        formatted += f"Possible Outcomes: {', '.join(classes)}\n"
        
        return formatted
    
    def handle_prediction(self):
        input_data = {}

        for col in self.predictor.categorical_columns:
            if col == 'Opponent':
                combo = self.combos[col]
                input_data[col] = combo.currentData() if combo.currentData() else combo.currentText().split(' ', 1)[-1]
            else:
                input_data[col] = self.combos[col].currentText()

        input_data['Score'] = self.score_spin.value()

        result, probs = self.predictor.predict(input_data)

        #resutl dikhanye k liye "="*50 ka matlab 50 spaces
        output = "=" * 50 + "\n"
        output += f"PREDICTION: {result.upper()}\n"
        output += "=" * 50 + "\n\n"

        # yeh agr id3 hoga to sirf win aur losss dikhaye ga, naive k liye % b dikhaye ga
        if self.predictor.algorithm == 'id3':
            output += "MODEL TYPE: ID3 Decision Tree (Deterministic)\n"
            output += "-" * 50 + "\n"
            output += f"Final Result: {result.upper()}\n"
        else:
            output += "MODEL TYPE: Naive Bayes (Probabilistic)\n"
            output += "-" * 50 + "\n"
            for res, p in probs.items():
                output += f"{res.capitalize():10s}: {p:.2%}\n"

        self.results_text.setText(output)

    