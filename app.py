import json
import random
import pickle

import numpy as np
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Burnout Support Chatbot", page_icon="💬")
st.title("💬 Burnout Support Chatbot")
st.caption(
    "An ML-based chatbot offering information and support around workplace burnout. "
    "This is not a substitute for professional mental health care."
)

ERROR_THRESHOLD = 0.25
OUT_OF_SCOPE_THRESHOLD = 0.55  # out_of_scope needs to be a lot more confident to fire mid-conversation

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


def keyword_safety_check(sentence):
    """Returns a crisis tag if the raw message matches a safety-critical phrase, else None."""
    s = sentence.lower()
    if any(kw in s for kw in HARM_OTHERS_KEYWORDS):
        return "crisis_harm_others"
    if any(kw in s for kw in SELF_HARM_KEYWORDS):
        return "crisis_self_harm"
    return None


# ----------------------------------------------------------------------
# Cached loading of NLTK data, model, and supporting files
# (runs once per app instance, not on every rerun)
# ----------------------------------------------------------------------
@st.cache_resource
def load_nltk_data():
    for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return WordNetLemmatizer()


@st.cache_resource
def load_chatbot_resources():
    model = load_model("burnout_chatbot_model.h5", compile=False)
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)
    return model, words, classes, intents


lemmatizer = load_nltk_data()
model, words, classes, intents = load_chatbot_resources()

# ----------------------------------------------------------------------
# Session state: conversation memory (per-user, per-browser-session)
# ----------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_tag" not in st.session_state:
    st.session_state.current_tag = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None


# ----------------------------------------------------------------------
# Core chatbot logic (mirrors the training notebook exactly)
# ----------------------------------------------------------------------
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
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            options = intent["responses"]
            choices = [r for r in options if r != st.session_state.last_response] or options
            response = random.choice(choices)
            st.session_state.last_response = response
            return response
    return "I'm sorry, I didn't quite understand that."


def get_response(sentence, intents_json):
    # Safety-critical keyword check runs before anything else, regardless of context.
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
        return (
            "I'm not quite sure I follow — could you tell me a bit more, or ask me "
            "something about burnout, stress, or how you're feeling?"
        )

    tag = predictions[0]["intent"]
    confidence = predictions[0]["probability"]

    if in_supportive_context and tag not in SUPPORTIVE_TAGS and confidence < OUT_OF_SCOPE_THRESHOLD:
        tag = "venting"

    st.session_state.current_tag = tag
    return pick_response(tag, intents_json)


# ----------------------------------------------------------------------
# Chat UI
# ----------------------------------------------------------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

if not st.session_state.chat_history:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi there. However you're doing today, I'm ready to listen. "
            "What would you like to talk about?"
        )

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    reply = get_response(user_input, intents)

    st.session_state.chat_history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)

with st.sidebar:
    st.subheader("About")
    st.write(
        "This chatbot uses a Bag-of-Words + feedforward neural network to classify "
        "user intent and respond with burnout-related support information."
    )
    st.write("It is not a substitute for professional mental health care.")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.current_tag = None
        st.session_state.last_response = None
        st.rerun()
