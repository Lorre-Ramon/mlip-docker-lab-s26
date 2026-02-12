from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the Wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize train test data
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# TODO: Create a RandomForestClassifier and train it on the training data
clf = RandomForestClassifier()
clf.fit(X_train_norm, y_train)

# Evaluate on test set
y_pred = clf.predict(X_test_norm)
print(f"Training complete. Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

# TODO: Save the trained model to the shared volume
model_dir = "/app/models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump( 
    value=clf
    , filename="/app/models/wine_model.pkl"
)

print("Model saved to /app/models/wine_model.pkl")
