import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'C:/Users/91960/Desktop/python_ws/bank/bank.csv'
data = pd.read_csv(file_path, delimiter=';')  # Semicolon used as delimiter

# Assign column names to the dataset
column_names = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
]
data.columns = column_names

# Display the first few rows of the dataset
print("Initial Dataset Overview:")
print(data.head())

# Save the cleaned dataset
data.to_csv('C:/Users/91960/Desktop/python_ws/bank/formatted_bank.csv', index=False)

# Re-load the formatted dataset for classification
data = pd.read_csv('C:/Users/91960/Desktop/python_ws/bank/formatted_bank.csv')

# Display the first few rows to confirm formatting
print("Formatted Dataset Preview:")
print(data.head())

# Encode categorical variables
label_encoders = {}
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Define feature set and target variable
X = data.drop(columns=['y'])
y = data['y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate and print the model's performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Detailed Classification Report:\n", classification_report(y_test, y_pred))