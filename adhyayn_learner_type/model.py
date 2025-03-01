import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("C:\\Users\\iamsa\\alt\\adhyayn_learner_type\\dataset\\my_csv.csv")

# Drop non-relevant columns (Gender & Age)
df = df.drop(columns=["Gender", "Age"], errors='ignore')

# Assume the last column is the learner type
X = df.iloc[:, :-1]  # Features (questions)
y = df.iloc[:, -1]   # Target (learner type)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Handle missing values (if any)
X = X.fillna(X.median())  # Replace NaN with column median

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "learner_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model trained and saved successfully!")
