import json
import random
import nltk
import numpy as np
import tensorflow as tf
import streamlit as st
from nltk.stem.lancaster import LancasterStemmer
import pickle
import os

# Download necessary NLTK data
nltk.download('punkt')

# Initialize stemmer
stemmer = LancasterStemmer()

# Load chatbot data
df = r"C:\Users\moham\Desktop\NTI_ETA_CU\Final Project\daily.json"

with open(df, "r", encoding="utf-8") as file:
    cb = json.load(file)

# Load processed data or preprocess
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
    words = sorted(set(words))
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

# Load or build the model
if os.path.exists("chatbot_model.h5"):
    model = tf.keras.models.load_model("chatbot_model.h5")
else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(len(training[0]),), activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(output[0]), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
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


# Streamlit App
st.title("🤖 Chatbot")
st.markdown("---")

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User Input
with st.container():
    st.subheader("Chat with Pandora!")
    user_input = st.text_input("You", placeholder="Type your message here...")

# Chat logic and response
if user_input:
    if user_input.lower() == "quit":
        st.write("Goodbye!")
    else:
        results = model.predict(np.array([bag_of_words(user_input, words)]))[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.2:  # Confidence threshold
            for tg in cb["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            response = random.choice(responses)
        else:
            response = "I'm not sure I understand. Can you rephrase?"
        
        # Add user and chatbot messages to the history
        st.session_state.messages.append({"user": user_input, "bot": response})

# Display chat history
for msg in st.session_state.messages:
    with st.container():
        st.write(f"**You:** {msg['user']}")
        st.markdown(f"**🤖 Chatbot:** {msg['bot']}")

st.markdown("---")
