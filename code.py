import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

# Deep Learning imports
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Load Dataset
df = pd.read_csv('mail_data.csv')
df = df.where(pd.notnull(df), '')

# Encode labels
df.loc[df['Category'] == 'spam', 'Category'] = 0
df.loc[df['Category'] == 'ham', 'Category'] = 1

X = df['Message']
y = df['Category'].astype('int')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# ===============================
# Classical ML Models (Baseline)
# ===============================
print("\nTraining Classical ML Models...\n")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = tfidf.fit_transform(X_train)
X_test_features = tfidf.transform(X_test)

joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train_features, y_train)
    y_pred = model.predict(X_test_features)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {"Accuracy": acc, "F1 Score": f1}
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

best_model_name = max(results, key=lambda x: results[x]['F1 Score'])
joblib.dump(models[best_model_name], f'{best_model_name}_model.pkl')
print(f"\nBest classical model saved: {best_model_name}")

# ===============================
# Deep Learning Model (LSTM)
# ===============================
print("\nBuilding Deep Learning Model...\n")

# Tokenization and Padding
vocab_size = 10000
max_length = 100
embedding_dim = 64

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Build LSTM Model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train Model
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"\nDeep Learning Model Accuracy: {accuracy:.4f}")

# Save Model
model.save('spam_detector_lstm.h5')

# ===============================
# Single Email Prediction
# ===============================
sample_email = ["Congratulations! You have won a prize. Claim now!"]
sample_seq = tokenizer.texts_to_sequences(sample_email)
sample_pad = pad_sequences(sample_seq, maxlen=max_length, padding='post')

prediction = model.predict(sample_pad)
print("\nSample Email Prediction (Deep Learning):")
print("Prediction:", "Ham" if prediction[0] > 0.5 else "Spam")
