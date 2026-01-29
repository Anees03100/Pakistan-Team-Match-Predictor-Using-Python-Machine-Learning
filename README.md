This project is a Desktop-based Machine Learning Application designed to predict the match results of the Pakistan National Cricket Team. Developed using Python, the application allows users to input specific match parameters and receive a predicted outcome based on historical data.

Key Features
Dual Algorithm Support: Users can switch between Naive Bayes (for probabilistic outcomes) and ID3 Decision Tree (for logic-based rule paths) to see how different models interpret the data.

Dynamic Data Processing: The system utilizes pandas and scikit-learn to process categorical match data such as Opponents, Venue, Recent Form, and Batting Order.

Interactive UI: Built with PyQt5, the interface features a dashboard where users can select match conditions via dropdowns and spin boxes.

Model Transparency: * Displays real-time Accuracy Scores and Confusion Matrices.

Provides a Decision Rule Viewer for the ID3 algorithm, allowing users to trace the specific "if-then" logic the model uses to reach a prediction.

Statistical Insights: Includes a built-in statistical summary of the dataset (mean, standard deviation, etc.) to help users understand the underlying data distribution.

Technical Stack
GUI: PyQt5

Data Analysis: Pandas, NumPy

Machine Learning: Scikit-learn (GaussianNB, DecisionTreeClassifier)

Visualization: Matplotlib
