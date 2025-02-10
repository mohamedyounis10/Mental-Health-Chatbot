### Mental Health Chatbot

**Mental Health Chatbot** is a conversational AI application designed to provide support and guidance to individuals seeking mental health assistance. The project is built using **Python** with **TensorFlow** for the chatbot's natural language processing and machine learning capabilities. It aims to deliver thoughtful and empathetic responses to user queries about mental health, fostering a supportive virtual environment.

---

### Key Features:
1. **Conversational AI**: Utilizes NLP techniques like tokenization and stemming to process user input and generate context-aware responses.
2. **Customizable Dataset**: The chatbot uses a JSON file (`daily.json`) to define intents, patterns, and responses, allowing for easy updates and scalability.
3. **Neural Network Model**: 
   - Input is processed into a bag-of-words representation.
   - A dense neural network with dropout layers ensures robust and accurate predictions for user queries.
4. **Streamlit-Based UI**: The project includes a **Streamlit app** to provide a clean and user-friendly graphical interface for interactions with the chatbot.

---

### Preview:
Below is a preview of the chatbot's user interface:

![UI Chatbot](https://github.com/user-attachments/assets/6d4f8230-e65a-4fa2-8d8f-3737c869b302)
 
---

### Technologies Used:
- **Python**: Core programming language.
- **TensorFlow**: For building and training the neural network model.
- **NLTK**: For natural language preprocessing (tokenization and stemming).
- **Streamlit**: For creating the UI.
- **JSON**: To store chatbot intents, patterns, and responses.

---

### How It Works:
1. **Dataset Processing**: Patterns from the dataset (`daily.json`) are tokenized, stemmed, and transformed into a bag-of-words representation for training.
2. **Model Training**: A neural network is trained to classify user inputs into predefined intents.
3. **User Interaction**: The chatbot predicts the intent of the user's query and responds with a matching response from the dataset. If the confidence is low, it asks for clarification.
4. **Streamlit UI**: The chatbot's functionality is integrated into a Streamlit application for an interactive and engaging user experience.

---

### Installation and Setup:
1. Clone this repository.
2. Install the required Python libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Start chatting with the bot and receive supportive mental health responses.

---

### Repository Structure:
- **`chatbot.py`**: Main chatbot logic and training script.
- **`daily.json`**: Dataset file containing intents, patterns, and responses.
- **`app.py`**: Streamlit application for the chatbot UI.
- **`data.pickle`**: Preprocessed training data for faster re-runs.
- **`chatbot_model.h5`**: Saved model file after training.

---

### Future Enhancements:
- Expanding the intents and responses to cover more mental health topics.
- Adding sentiment analysis for more empathetic responses.
- Integrating external APIs for mental health resources and helpline numbers.
