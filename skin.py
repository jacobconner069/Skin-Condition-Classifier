import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import medmnist
from medmnist import DermaMNIST
import time
#Part 1: Data Exploration and Preprocessing

np.random.seed(42)
torch.manual_seed(42)
# Load the dataset
training_data = DermaMNIST(split='train', download=True, size=28)
validation_data = DermaMNIST(split='val', download=True, size=28)
testing_data = DermaMNIST(split='test', download=True, size=28)
print("Loaded the dataset")

# Convert the datasets to numpy arrays
x_train, y_train = training_data.imgs, training_data.labels.squeeze()
x_val, y_val = validation_data.imgs, validation_data.labels.squeeze()
x_test, y_test = testing_data.imgs, testing_data.labels.squeeze()

print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
print(f"Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}")

class_names = ["Melanocytic nevi", "Melanoma", "Benign keratosis", "Basal cell carcinoma", "Actinic keratoses", "Vascular lesions", "Dermatofibroma"]
num_classes = len(class_names)

# Normalize the pixel values to the range [0, 1]
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Display sample images
fig, axes = plt.subplots(1, 7, figsize=(15, 3))
fig.suptitle('Sample Images from Each Class', fontsize=16)
for i in range(num_classes):
    idx = np.where(y_train ==i)[0][0]
    axes[i].imshow(x_train[idx])
    axes[i].set_title(class_names[i])
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("sample_images.png")
plt.close()
print("Sample images saved as 'sample_images.png'")

# Class distribution

plt.figure(figsize=(10, 6))
plt.bar(class_names, [np.sum(y_train == i) for i in range(num_classes)])
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
print("Class distribution plot saved as 'class_distribution.png'")
plt.savefig("class_distribution.png")
plt.close()

#Part 2: Logistic Regression 
x_train_flat = x_train.reshape(len(x_train), -1).astype(np.float32)
x_val_flat = x_val.reshape(len(x_val), -1).astype(np.float32)
x_test_flat = x_test.reshape(len(x_test), -1).astype(np.float32)
print("Data flattened for logistic regression")

# Train a logistic regression model
c_values = [0.001, 0.01, 0.1, 1, 10]
best_f1 = 0
best_c = None
model = LogisticRegression(max_iter=5000, solver='lbfgs', class_weight='balanced', warm_start=True)
for c in c_values:
    model.set_params(C=c)
    model.fit(x_train_flat, y_train)
    pred = model.predict(x_val_flat)
    val_f1 = f1_score(y_val, pred, average='macro')
    print(f"C={c}: Validation F1 Score = {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_c = c

print(f"Best C value: {best_c}, Best F1 Score: {best_f1:.4f}")

# Evaluate the best model on the test set
def evaluate_metrics(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division = 0)
    recall = recall_score(y_true, y_pred, average="macro",zero_division = 0)
    f1 = f1_score(y_true, y_pred, average = "macro", zero_division = 0)
    print(f"\n [{name}] Accuracy={accuracy: .4f} Precision={precision: .4f} Recall={recall:.4f} F1={f1:.4f}")
    return accuracy, precision, recall, f1

best_model = LogisticRegression(C=best_c, max_iter=5000, solver='lbfgs', class_weight='balanced')
lr_start = time.time()
best_model.fit(x_train_flat, y_train)
lr_time = time.time() - lr_start
val_pred = best_model.predict(x_val_flat)
test_pred = best_model.predict(x_test_flat)
val_metrics = evaluate_metrics("LR Val", y_val, val_pred)
test_metrics = evaluate_metrics("LR Test", y_test, test_pred)

cm = confusion_matrix(y_test, test_pred)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
ax.set_xticklabels([f"C{i}" for i in range(num_classes)])
ax.set_yticklabels([f"C{i}" for i in range(num_classes)])
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_title("Logistic Regression – Confusion Matrix (Test)", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png", dpi=120, bbox_inches="tight")
plt.close()
print("Confusion matrix saved as confusion_matrix_lr.png")

#Part 3 Neural Network
def tens(x,y):
    X = torch.tensor(x.reshape(len(x), -1), dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X,Y)

train_ds = tens(x_train, y_train)
val_ds = tens(x_val, y_val)
test_ds = tens(x_test, y_test)

train_load = DataLoader(train_ds, batch_size=128, shuffle=True)
val_load = DataLoader(val_ds, batch_size=128)
test_load = DataLoader(test_ds, batch_size=128)

input_size = 28*28*3
hidden1 = 512
hidden2 = 256
output = num_classes

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size,hidden1), nn.BatchNorm1d(hidden1), 
        nn.ReLU(), nn.Dropout(0.4), nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2),
        nn.ReLU(), nn.Dropout(0.4), nn.Linear(hidden2,output))

    def forward(self, x):
        return self.net(x)
    
device = torch.device("cpu")
modell = MLP().to(device)
print(modell)

class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modell.parameters(), lr=5e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

EPOCHS = 30
train_losses, val_losses = [], []
train_accs, val_accs = [], []

nn_start = time.time()
for epoch in range(1, EPOCHS + 1):
    modell.train()
    epoch_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_load:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = modell(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += len(yb)
    train_losses.append(epoch_loss / total)
    train_accs.append(correct/total)

    modell.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_load:
            xb, yb = xb.to(device), yb.to(device)
            logits = modell(xb)
            v_loss    += criterion(logits, yb).item() * len(xb)
            v_correct += (logits.argmax(1) == yb).sum().item()
            v_total   += len(yb)
    val_losses.append(v_loss / v_total)
    val_accs.append(v_correct / v_total)
 
    scheduler.step(val_losses[-1])
 
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>2}/{EPOCHS}  "
              f"Train Loss={train_losses[-1]:.4f} Acc={train_accs[-1]:.4f}  "
              f"Val Loss={val_losses[-1]:.4f} Acc={val_accs[-1]:.4f}")
    nn_time = time.time() - nn_start
 
# ── 4e. Evaluation ──
def predict_all(loader):
    all_preds, all_labels = [], []
    modell.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = modell(xb).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)
 
nn_val_pred,  _ = predict_all(val_load)
nn_test_pred, _ = predict_all(test_load)
 
nn_val_metrics  = evaluate_metrics("MLP Val", y_val,  nn_val_pred)
nn_test_metrics = evaluate_metrics("MLP Test", y_test, nn_test_pred)
 
# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train"); ax1.plot(val_losses, label="Val")
ax1.set_title("Loss Curves"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(train_accs, label="Train"); ax2.plot(val_accs, label="Val")
ax2.set_title("Accuracy Curves"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.suptitle("MLP Training History", fontweight="bold")
plt.tight_layout()
plt.savefig("nn_training_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: nn_training_curves.png")
 
# Confusion matrix – MLP
cm_nn = confusion_matrix(y_test, nn_test_pred)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_nn, interpolation="nearest", cmap="Greens")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
ax.set_xticklabels([f"C{i}" for i in range(num_classes)])
ax.set_yticklabels([f"C{i}" for i in range(num_classes)])
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm_nn[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm_nn[i, j] > cm_nn.max()/2 else "black")
ax.set_title("MLP Neural Network – Confusion Matrix (Test)", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_nn.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrix_nn.png")
 
#Part 4: Comparison Table
metrics_names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "Training Time (s)"]
lr_row  = list(test_metrics) + [round(lr_time, 2)]
nn_row  = list(nn_test_metrics) + [round(nn_time, 2)]
 
df_compare = pd.DataFrame(
    {
        "Metric"              : metrics_names,
        "Logistic Regression" : lr_row,
        "MLP Neural Network"  : nn_row,
    }
)
print(df_compare.to_string(index=False))
df_compare.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")
 
print("\nDone. All outputs saved.")