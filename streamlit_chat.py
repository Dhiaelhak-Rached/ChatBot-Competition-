import random
import json
import torch
import streamlit as st
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load intents and model data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]   
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Investo"

st.set_page_config(
    page_title="Chat with Investo",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Chat with Investo")

user_input = st.text_input("You: ", "")
output_area = st.empty()

if st.button("Send"):
    with st.spinner("Thinking..."):
        if user_input.lower() == "quit":
            st.success(f"{bot_name}: Goodbye!")
        else:
            # Check if the input starts with a greeting keyword
            if user_input.lower().startswith(("hello", "hi", "hey")):
                # Extract additional information, e.g., the name
                input_tokens = tokenize(user_input)
                name = input_tokens[1] if len(input_tokens) > 1 else None
                st.write(f"{bot_name}: Hello, {name}! How can I assist you today?")
            else:
                # If the input doesn't match a greeting, proceed with the existing logic
                sentence = tokenize(user_input)
                X = bag_of_words(sentence, all_words)
                X = X.reshape(1, X.shape[0])
                X = torch.from_numpy(X).to(device)

                output = model(X)
                _, predicted = torch.max(output, dim=1)

                tag = tags[predicted.item()]

                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()]
                if prob.item() > 0.75:
                    for intent in intents['intents']:
                        if tag == intent["tag"]:
                            st.write(f"{bot_name}: {random.choice(intent['responses'])}")
                else:
                    st.write(f"{bot_name}: I do not understand...")
