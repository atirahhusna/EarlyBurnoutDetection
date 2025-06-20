import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and clean data
df = pd.read_csv("balanced_mental_health_dataset.csv")
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
columns_to_drop = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'country' in col.lower()]
df = df.drop(columns=columns_to_drop)

# Label encoding
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

X = df.drop("Growing_Stress", axis=1)
y = df["Growing_Stress"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save model
model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("trained_model.sav", "wb") as f:
    pickle.dump((model, X.columns.tolist(), label_encoders), f)

