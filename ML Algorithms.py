import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = "/Users/wolfmigo/Downloads/wine/wine.csv"
columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
           'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
           'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

wine_data = pd.read_csv(data_path, names=columns)

# Split features and labels
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train classifiers and obtain confusion matrices
confusion_matrices = {}
accuracies = {}
for clf_name, clf in classifiers.items():
    if clf_name == 'Gradient Boosting':
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    confusion_matrices[clf_name] = confusion_matrix(y_test, y_pred)
    accuracies[clf_name] = accuracy_score(y_test, y_pred)

# Plot confusion matrices
plt.figure(figsize=(15, 10))
for i, (clf_name, matrix) in enumerate(confusion_matrices.items(), 1):
    plt.subplot(2, 3, i)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", cbar=True)  # Set cbar=True for color bar
    plt.title(f"Confusion Matrix for {clf_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
plt.tight_layout()
plt.show()

# Print accuracies
print("Accuracies:")
for clf_name, acc in accuracies.items():
    print(f"{clf_name}: {acc}")

# Compare performances
best_classifier = max(accuracies, key=accuracies.get)
print(f"\nBest performing classifier: {best_classifier} with accuracy of {accuracies[best_classifier]}")
