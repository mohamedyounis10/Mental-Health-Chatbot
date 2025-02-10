import json
import random
import nltk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import pickle
import os
from time import sleep
from nltk.stem.lancaster import LancasterStemmer
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
# Download necessary NLTK data
nltk.download('punkt')

# Initialize stemmer
stemmer = LancasterStemmer()

# Load chatbot data
df = r"C:\Users\seifk\Desktop\daily.json"

with open(df, "r", encoding="utf-8") as file:
    cb = json.load(file)

# Try to load previous training data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Process intents from JSON
    for intent in cb["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(set(words))  # Remove duplicates
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            bag.append(1 if w in wrds else 0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Convert to numpy arrays
    training = np.array(training)
    output = np.array(output)

    # Save processed data
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Build Neural Network using tf.keras
model = Sequential([
    Dense(128, input_shape=(len(training[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(output[0]), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(training, output, epochs=1000, batch_size=8, verbose=1)

# Save the model
model.save("chatbot_model.h5")


# Function to Convert Input to Bag of Words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s.lower())
    s_words = [stemmer.stem(word) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


# Chatbot Function
def chat():
    print("Hi! How can I assist you today :-) ? (Type 'quit' to exit)")
    
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Goodbye!")
            break
        
        results = model.predict(np.array([bag_of_words(inp, words)]))[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.2:  # Confidence threshold
            for tg in cb["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            sleep(1)
            print(random.choice(responses))
        else:
            print("I'm not sure I understand. Can you rephrase?")


# Run the Chatbot
chat()
