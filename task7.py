import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Setup Output Folder
OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear SVM
linear_svm = SVC(kernel='linear', C=1, random_state=42)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

linear_acc = accuracy_score(y_test, y_pred_linear)
linear_report = classification_report(y_test, y_pred_linear)
linear_cm = confusion_matrix(y_test, y_pred_linear)

with open(os.path.join(OUTPUT_DIR, "linear_results.txt"), "w", encoding="utf-8") as f:
    f.write("Linear SVM Results\n")
    f.write(f"Accuracy: {linear_acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(linear_report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(linear_cm))

# RBF Kernel SVM
rbf_svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

rbf_acc = accuracy_score(y_test, y_pred_rbf)
rbf_report = classification_report(y_test, y_pred_rbf)
rbf_cm = confusion_matrix(y_test, y_pred_rbf)

with open(os.path.join(OUTPUT_DIR, "rbf_results.txt"), "w", encoding="utf-8") as f:
    f.write("RBF Kernel SVM Results\n")
    f.write(f"Accuracy: {rbf_acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(rbf_report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(rbf_cm))

# Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 0.01],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, cv=5, refit=True)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_params = grid.best_params_
best_cv_score = grid.best_score_
best_test_acc = accuracy_score(y_test, best_model.predict(X_test))

with open(os.path.join(OUTPUT_DIR, "best_model_info.txt"), "w", encoding="utf-8") as f:
    f.write("Best Model Found by GridSearchCV\n")
    f.write(f"Best Params: {best_params}\n")
    f.write(f"Best CV Score: {best_cv_score:.4f}\n")
    f.write(f"Test Accuracy: {best_test_acc:.4f}\n")

# Visualization (2D)
X_vis = X_train[:, :2]
y_vis = y_train

svm_vis = SVC(kernel='rbf', C=1, gamma='scale')
svm_vis.fit(X_vis, y_vis)

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.coolwarm)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("SVM Decision Boundary (RBF Kernel)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig(os.path.join(OUTPUT_DIR, "decision_boundary.png"))
plt.close()

# Cross-validation Scores
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
with open(os.path.join(OUTPUT_DIR, "cross_validation.txt"), "w", encoding="utf-8") as f:
    f.write("Cross-validation scores:\n")
    f.write(str(cv_scores))
    f.write(f"\nMean CV Accuracy: {np.mean(cv_scores):.4f}")

# Summary File
with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("SVM Binary Classification Summary\n")
    f.write(f"Linear Accuracy: {linear_acc:.4f}\n")
    f.write(f"RBF Accuracy: {rbf_acc:.4f}\n")
    f.write(f"Best Params: {best_params}\n")
    f.write(f"Best CV Score: {best_cv_score:.4f}\n")
    f.write(f"Test Accuracy (Best Model): {best_test_acc:.4f}\n")
    f.write(f"Artifacts saved in: {os.path.abspath(OUTPUT_DIR)}\n")

print("\nAll artifacts saved successfully in the 'artifacts/' folder!")