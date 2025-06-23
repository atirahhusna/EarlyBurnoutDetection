import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("/content/sample_data/balanced_mental_health_dataset.csv")

# Fill missing values with mode
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Drop timestamp, date, and country-related columns
columns_to_drop = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'country' in col.lower()]
df.drop(columns=columns_to_drop, inplace=True)

# Encode categorical features
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Define features and target
X = df.drop('Growing_Stress', axis=1)
y = df['Growing_Stress']

# 1️⃣ Accuracy before Feature Selection
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X, y, test_size=0.2, random_state=42)
model_base = RandomForestClassifier(random_state=42)
model_base.fit(X_train_base, y_train_base)
y_pred_base = model_base.predict(X_test_base)
baseline_accuracy = accuracy_score(y_test_base, y_pred_base)
print("🟡 Accuracy (Before Feature Selection): {:.4f}".format(baseline_accuracy))

# 2️⃣ Feature Importances
model_full = RandomForestClassifier(random_state=42)
model_full.fit(X, y)
importances = model_full.feature_importances_
importance_df = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=importance_df.values, y=importance_df.index, palette='magma')
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Select features above threshold
threshold = 0.05
selected_features = importance_df[importance_df > threshold].index.tolist()
importance_df[selected_features].to_csv("rf_selected_features.csv", header=True)

print("\n✅ Selected Features (Importance > 0.05):")
print(importance_df[selected_features])

# 3️⃣ Cross-Validation
X_selected = X[selected_features]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_cv = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model_cv, X_selected, y, cv=skf, scoring='accuracy')

print("\n🔁 5-Fold Cross-Validation Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f" - Fold {i}: {score:.4f}")
print("📊 Mean CV Accuracy: {:.4f}".format(cv_scores.mean()))

# 4️⃣ GridSearchCV to tune n_estimators
print("\n🔍 Tuning n_estimators using GridSearchCV...")
param_grid = {'n_estimators': [50, 100, 150, 200]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=skf, scoring='accuracy')
grid_search.fit(X_selected, y)
best_model = grid_search.best_estimator_

print("✅ Best n_estimators:", grid_search.best_params_['n_estimators'])

# 5️⃣ Evaluate final model
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)
best_model.fit(X_train_sel, y_train_sel)
y_pred_sel = best_model.predict(X_test_sel)
final_accuracy = accuracy_score(y_test_sel, y_pred_sel)

print("\n✅ Final Accuracy (After Feature Selection + GridSearchCV): {:.4f}".format(final_accuracy))
print("\n📋 Classification Report:\n", classification_report(y_test_sel, y_pred_sel))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_sel, y_pred_sel), annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix - Random Forest (Optimized)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 6️⃣ Summary
print("\n📌 Accuracy Summary:")
print(f" - Before Feature Selection: {baseline_accuracy:.4f}")
print(f" - Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f" - Final Test Accuracy: {final_accuracy:.4f}")

# 7️⃣ Save the final model
import pickle

with open("trained_model.sav", "wb") as f:
    pickle.dump((best_model, X.columns.tolist(), label_encoders), f)

print("✅ Model saved as 'trained_model.sav'")

