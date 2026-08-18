import json
import random
import pickle

import numpy as np
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Burnout Support Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Burnout Support Chatbot")
st.caption("A basic AI chatbot for burnout and stress-related support.")


@st.cache_resource
def setup_nltk():
    for package in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass
    return WordNetLemmatizer()


lemmatizer = setup_nltk()


@st.cache_resource
def load_chatbot():
    model = load_model("burnout_chatbot_model.h5")

    with open("words.pkl", "rb") as f:
        words = pickle.load(f)

    with open("classes.pkl", "rb") as f:
        classes = pickle.load(f)

    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    return model, words, classes, intents


model, words, classes, intents = load_chatbot()

ERROR_THRESHOLD = 0.25
OUT_OF_SCOPE_THRESHOLD = 0.55

SUPPORTIVE_TAGS = {
    "venting", "burnout_definition", "burnout_symptoms", "burnout_causes",
    "burnout_vs_stress", "burnout_stages", "self_help_strategies",
    "early_detection", "work_life_balance", "talking_to_manager",
    "student_burnout", "remote_work_burnout", "burnout_statistics",
    "breathing_exercise", "time_management_tips", "sleep_tips",
    "seek_professional_help", "crisis_support", "mindfulness_detail",
    "exercise_detail", "boundaries_detail", "breaks_detail",
    "communication_detail", "crisis_self_harm", "crisis_harm_others",
    "crisis_passive_ideation", "low_self_worth", "anxious",
    "reluctant_to_share", "bot_limitation", "casual_ack",
    "conversation_repair", "topic_change", "isolation"
}

SELF_HARM_KEYWORDS = [
    "wanna die", "want to die", "want to end my life", "kill myself",
    "end my life", "don't want to live", "want to end it all",
    "hurt myself", "thinking about suicide", "feel like ending it",
    "end it all", "sleep until i don't wake up",
    "sleep until not wake up", "wish i wouldn't wake up",
    "never wake up", "don't want to wake up"
]

HARM_OTHERS_KEYWORDS = [
    "want to kill him", "want to kill her", "want to kill them",
    "want to kill my", "want to hurt someone", "want to hurt him",
    "want to hurt her", "attacking someone", "kill him", "kill her"
]

conversation_context = {"current_tag": None}
last_response_used = {"text": None}


def keyword_safety_check(sentence):
    s = sentence.lower()
    if any(kw in s for kw in HARM_OTHERS_KEYWORDS):
        return "crisis_harm_others"
    if any(kw in s for kw in SELF_HARM_KEYWORDS):
        return "crisis_self_harm"
    return None


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]


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

    result = model.predict(np.array([bow]), verbose=0)[0]

    results = [
        [i, probability]
        for i, probability in enumerate(result)
        if probability > ERROR_THRESHOLD
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    return [
        {"intent": classes[item[0]], "probability": float(item[1])}
        for item in results
    ]


def pick_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            options = intent["responses"]
            choices = [
                response for response in options
                if response != last_response_used["text"]
            ] or options

            response = random.choice(choices)
            last_response_used["text"] = response
            return response

    return "I'm sorry, I didn't quite understand that."


def get_response(sentence):
    # Safety-critical keyword check happens before ML prediction.
    crisis_tag = keyword_safety_check(sentence)
    if crisis_tag:
        conversation_context["current_tag"] = crisis_tag
        return pick_response(crisis_tag)

    predictions = predict_class(sentence)
    in_supportive_context = (
        conversation_context["current_tag"] in SUPPORTIVE_TAGS
    )

    if not predictions:
        if in_supportive_context:
            return pick_response("venting")

        conversation_context["current_tag"] = None
        return (
            "I'm not quite sure I follow — could you tell me a bit more, "
            "or ask me something about burnout, stress, or how you're feeling?"
        )

    tag = predictions[0]["intent"]
    confidence = predictions[0]["probability"]

    if (
        in_supportive_context
        and tag not in SUPPORTIVE_TAGS
        and confidence < OUT_OF_SCOPE_THRESHOLD
    ):
        tag = "venting"

    conversation_context["current_tag"] = tag
    return pick_response(tag)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi there. I'm the Burnout Support Bot. "
                "I'm here to listen and help you understand burnout. "
                "What would you like to talk about?"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_response(prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)


with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot uses a Bag-of-Words neural-network intent classifier "
        "with context-awareness and a separate safety keyword layer."
    )
    st.warning(
        "This chatbot is not a therapist or emergency service. "
        "For serious or immediate danger, seek professional or emergency help."
    )
