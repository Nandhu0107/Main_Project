import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# ======================================
# 1️⃣ Load Saved Multimodal Model
# ======================================
model = tf.keras.models.load_model("multimodal_traffic_model.keras")

# ======================================
# 2️⃣ Load Test Data
# (Make sure these match your saved test arrays)
# ======================================
X_test_images = np.load("X_images.npy")
X_test_structured = np.load("X_structured_aligned.npy")
y_test = np.load("y_labels.npy")

# If labels are one-hot encoded:
y_true = np.argmax(y_test, axis=1)

# ======================================
# 3️⃣ Make Predictions
# ======================================
y_pred_probs = model.predict([X_test_images, X_test_structured])
y_pred = np.argmax(y_pred_probs, axis=1)

# ======================================
# 4️⃣ Basic Metrics
# ======================================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n===== MULTIMODAL MODEL PERFORMANCE =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# ======================================
# 5️⃣ Classification Report
# ======================================
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_true, y_pred, target_names=["Emergency", "Normal"]))

# ======================================
# 6️⃣ Confusion Matrix
# ======================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=["Emergency", "Normal"],
            yticklabels=["Emergency", "Normal"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ======================================
# 7️⃣ ROC Curve (Binary Classification)
# ======================================
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

print("\nAUC Score:", roc_auc)

# ======================================
# 8️⃣ Save Results to File
# ======================================
with open("final_results.txt", "w") as f:
    f.write("Accuracy: " + str(accuracy) + "\n")
    f.write("Precision: " + str(precision) + "\n")
    f.write("Recall: " + str(recall) + "\n")
    f.write("F1 Score: " + str(f1) + "\n")
    f.write("AUC: " + str(roc_auc) + "\n")

print("\nEvaluation completed and results saved!")