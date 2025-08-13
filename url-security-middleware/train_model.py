import pandas as pd
import numpy as np
import re
import os
import pickle
import random
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report



from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# 📥 Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("malicious_phish.csv")

# ✅ Use correct column names
if "url" in df.columns and "type" in df.columns:
    df = df[["url", "type"]].dropna()
    df.columns = ["url", "label"]
else:
    raise Exception("❌ Dataset does not contain expected columns ['url', 'type'].")

df["label"] = df["label"].str.lower()

# --- Robust synthetic data augmentation for all classes ---
from url_generator import URLGenerator

def generate_benign_urls(n=12000):
    gen = URLGenerator(seed=123)
    urls = set()
    tlds = [".com", ".org", ".net", ".edu", ".gov", ".in", ".io", ".ai", ".co", ".info", ".xyz", ".shop"]
    for _ in range(n):
        # Use generator, then add tld/randomization
        url = gen.generate_valid_url()
        # Randomly swap TLD
        if random.random() < 0.3:
            parts = url.split('.')
            if len(parts) > 1:
                url = '.'.join(parts[:-1]) + random.choice(tlds)
        # Add IP-based benign
        if random.random() < 0.05:
            url = f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/home"
        urls.add(url)
    return list(urls)

def generate_malicious_urls(n=12000):
    gen = URLGenerator(seed=456)
    urls = set()
    for _ in range(n):
        url = gen.generate_invalid_url()
        # Add encoded/junk/edge
        if random.random() < 0.1:
            url += "%3Csvg/onload=alert(1)%3E"
        if random.random() < 0.05:
            url = f"ftp://malicious-{random.randint(1000,9999)}.tk/{random.randint(1,99999)}"
        urls.add(url)
    return list(urls)

def generate_edge_case_urls(n=10000):
    urls = set()
    for _ in range(n):
        choice = random.choice([0,1,2,3,4])
        if choice == 0:
            # IP URL
            url = f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
        elif choice == 1:
            # FTP
            url = f"ftp://randomsite{random.randint(1,9999)}.xyz/file{random.randint(1,99999)}.txt"
        elif choice == 2:
            # Encoded
            url = f"http://example.com/%3Csvg/onload=alert(1)%3E"
        elif choice == 3:
            # Port
            url = f"http://example.com:{random.randint(1025,65535)}/index"
        else:
            # Partial URL
            url = f"www.site{random.randint(1,9999)}.com/path{random.randint(1,999)}"
        urls.add(url)
    return list(urls)

def generate_not_a_url_samples(n=12000):
    samples = []
    for _ in range(n):
        length = random.randint(8, 40)
        choice = random.choice([0, 1, 2, 3])
        if choice == 0:
            s = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        elif choice == 1:
            s = ' '.join([''.join(random.choices(string.ascii_lowercase, k=random.randint(2, 8))) for _ in range(random.randint(2, 7))])
        elif choice == 2:
            s = ''.join(random.choices(string.punctuation + string.digits, k=length))
        else:
            s = 'http://' + ''.join(random.choices(string.ascii_lowercase, k=random.randint(1, 5)))
        samples.append(s)
    return samples

print("\n🔬 Generating robust synthetic samples for all classes...")
# Generate a large, diverse, and challenging dataset for each class
benign_urls = generate_benign_urls(15000)
malicious_urls = generate_malicious_urls(15000)
edge_urls = generate_edge_case_urls(12000)
not_a_url_samples = generate_not_a_url_samples(15000)

# Assign labels
benign_df = pd.DataFrame({"url": benign_urls, "label": "benign"})
malicious_df = pd.DataFrame({"url": malicious_urls, "label": "phishing"})
edge_df = pd.DataFrame({"url": edge_urls, "label": "edge_case"})
not_a_url_df = pd.DataFrame({"url": not_a_url_samples, "label": "not_a_url"})

# Combine with real data
frames = [df, benign_df, malicious_df, edge_df, not_a_url_df]
df = pd.concat(frames, ignore_index=True)

# Shuffle and balance dataset
print(f"[INFO] Dataset shape before balancing: {df.shape}")
min_count = min(df['label'].value_counts())
balanced_df = df.groupby('label').sample(n=min_count, random_state=42).reset_index(drop=True)
print(f"[INFO] Dataset shape after balancing: {balanced_df.shape}")
df = balanced_df

# Balance classes (downsample if needed)
min_class_count = min(df["label"].value_counts())
df = df.groupby("label").sample(n=min_class_count, random_state=42, replace=False)

print("\n📊 Class distribution (after robust augmentation):")
print(df["label"].value_counts())

# 🔐 Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# 🔧 Hyperparameters
MAX_LEN = 200
VOCAB_SIZE = 10000
EMBED_DIM = 128

# 🔡 Tokenize URLs
tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=True, lower=True)
tokenizer.fit_on_texts(df["url"])
sequences = tokenizer.texts_to_sequences(df["url"])
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = df["label_encoded"].values

# 🧪 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)

# Class weights for balanced training
class_weight_dict = dict(zip(np.unique(y_train), compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

# Build model
model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=128,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate
print("\n✅ Evaluating model...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Print class-wise accuracy
from collections import Counter
def classwise_accuracy(y_true, y_pred, labels):
    accs = {}
    for label in labels:
        idx = (y_true == label)
        acc = np.mean(y_pred[idx] == label)
        accs[label] = acc
    return accs
labels = np.unique(y_test)
accs = classwise_accuracy(y_test, y_pred, labels)
print("\nClass-wise accuracy:")
for label in labels:
    print(f"  {label_encoder.inverse_transform([label])[0]}: {accs[label]*100:.2f}%")

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 💾 Save everything
os.makedirs("saved_models", exist_ok=True)
print("\n💾 Saving model and preprocessing tools...")
model.save("saved_models/url_cnn_lstm_model.keras")
with open("saved_models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("saved_models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n✅ Training complete. Model and tools saved in 'saved_models/'")
