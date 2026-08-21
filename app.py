import json
import random
import pickle
import numpy as np
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Set random seeds for reproducibility
random.seed(42)    
np.random.seed(42)
tf.random.set_seed(42)

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')   
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Load Dataset
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Preprocess Text
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

# Build Training Data (Bag of Words)
training = []
output_empty = [0] * len(classes)

for word_list, tag in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in word_list]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([row[0] for row in training])
train_y = np.array([row[1] for row in training])

# Build and Train the Neural Network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(train_y[0]), activation='softmax'))

adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print("Training the model...")
model.fit(train_x, train_y, epochs=50, batch_size=5, verbose=1)

# Save the model and supporting files
model.save('burnout_chatbot_model.h5')
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)
print("Model and supporting files saved successfully.")

# Model Evaluation
print("\nRetraining on 80% of the data for actual evaluation...")
X = train_x
y = np.argmax(train_y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

eval_model = Sequential()
eval_model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
eval_model.add(Dropout(0.2))
eval_model.add(Dense(64, activation='relu'))
eval_model.add(Dropout(0.2))
eval_model.add(Dense(len(classes), activation='softmax'))
eval_model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

eval_model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=0)
y_pred = np.argmax(eval_model.predict(X_test, verbose=0), axis=1)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("="*50)
print(f"Accuracy:  {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"F1 Score:  {f1:.2f}")
print("="*50)