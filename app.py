import streamlit as st
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Burnout Support Chatbot", page_icon="💙")
st.title("💙 Burnout Support Chatbot")
st.write("I'm here to listen and help you manage occupational and academic burnout. What's on your mind today?")


@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
download_nltk_data()

lemmatizer = WordNetLemmatizer()

# Load pretrained model
@st.cache_resource
def load_chatbot_data():
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('burnout_chatbot_model.keras')
    return intents, words, classes, model

intents, words, classes, model = load_chatbot_data()

# Configuration and Thresholds
ERROR_THRESHOLD = 0.25
OUT_OF_SCOPE_THRESHOLD = 0.55 

SUPPORTIVE_TAGS = {
    "venting", "burnout_definition", "burnout_symptoms", "burnout_causes",
    "burnout_vs_stress", "burnout_stages", "self_help_strategies", "early_detection",
    "work_life_balance", "talking_to_manager", "student_burnout", "remote_work_burnout",
    "burnout_statistics", "breathing_exercise", "time_management_tips", "sleep_tips",
    "seek_professional_help", "crisis_support", "mindfulness_detail", "exercise_detail",
    "boundaries_detail", "breaks_detail", "communication_detail",
    "crisis_self_harm", "crisis_harm_others", "crisis_passive_ideation",
    "low_self_worth", "anxious", "reluctant_to_share", "bot_limitation", "casual_ack",
    "conversation_repair", "topic_change", "isolation"
}

SELF_HARM_KEYWORDS = [
    "wanna die", "want to die", "want to end my life", "kill myself", "end my life",
    "don't want to live", "want to end it all", "hurt myself", "thinking about suicide",
    "feel like ending it", "end it all", "sleep until i don't wake up", "sleep until not wake up", 
    "wish i wouldn't wake up", "never wake up", "don't want to wake up"
]

HARM_OTHERS_KEYWORDS = [
    "want to kill him", "want to kill her", "want to kill them", "want to kill my",
    "want to hurt someone", "want to hurt him", "want to hurt her", "attacking someone",
    "kill him", "kill her"
]

# Initialize session state for chat history and context
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_tag" not in st.session_state:
    st.session_state.current_tag = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None

def keyword_safety_check(sentence):
    s = sentence.lower()
    if any(kw in s for kw in HARM_OTHERS_KEYWORDS):
        return "crisis_harm_others"
    if any(kw in s for kw in SELF_HARM_KEYWORDS):
        return "crisis_self_harm"
    return None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for sw in sentence_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = 1
    return np.array(bag), sum(bag)

def predict_class(sentence):
    bow, matched_word_count = bag_of_words(sentence)
    if matched_word_count == 0:
        return []
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": float(r[1])} for r in results]

def pick_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            options = intent['responses']
            choices = [r for r in options if r != st.session_state.last_response] or options
            response = random.choice(choices)
            st.session_state.last_response = response
            return response
    return "I'm sorry, I didn't quite understand that."

def get_response(sentence, intents_json):
    crisis_tag = keyword_safety_check(sentence)
    if crisis_tag:
        st.session_state.current_tag = crisis_tag
        return pick_response(crisis_tag, intents_json)

    predictions = predict_class(sentence)
    in_supportive_context = st.session_state.current_tag in SUPPORTIVE_TAGS

    if not predictions:
        if in_supportive_context:
            return pick_response("venting", intents_json)
        st.session_state.current_tag = None
        return "I'm not quite sure I follow — could you tell me a bit more, or ask me something about burnout, stress, or how you're feeling?"

    tag = predictions[0]['intent']
    confidence = predictions[0]['probability']
    
    if in_supportive_context and tag not in SUPPORTIVE_TAGS and confidence < OUT_OF_SCOPE_THRESHOLD:
        tag = "venting"

    st.session_state.current_tag = tag
    return pick_response(tag, intents_json)

# Streamlit Chat UI
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    bot_response = get_response(prompt, intents)
    
    # Display bot response in chat message container
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})