import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load datasets and trained models
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('burnout_chatbot_model.h5')

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
            choices = [r for r in options if r != last_response_used["text"]] or options
            response = random.choice(choices)
            last_response_used["text"] = response
            return response
    return "I'm sorry, I didn't quite understand that."

def get_response(sentence, intents_json):
    crisis_tag = keyword_safety_check(sentence)
    if crisis_tag:
        conversation_context["current_tag"] = crisis_tag
        return pick_response(crisis_tag, intents_json)

    predictions = predict_class(sentence)
    in_supportive_context = conversation_context["current_tag"] in SUPPORTIVE_TAGS

    if not predictions:
        if in_supportive_context:
            return pick_response("venting", intents_json)
        conversation_context["current_tag"] = None
        return "I'm not quite sure I follow — could you tell me a bit more, or ask me something about burnout, stress, or how you're feeling?"

    tag = predictions[0]['intent']
    confidence = predictions[0]['probability']
    
    if in_supportive_context and tag not in SUPPORTIVE_TAGS and confidence < OUT_OF_SCOPE_THRESHOLD:
        tag = "venting"

    conversation_context["current_tag"] = tag
    return pick_response(tag, intents_json)

# Start the Chatbot Terminal Interface
if __name__ == "__main__":
    print("This is Burnout Support Chatbot! Type 'quit' to exit.\n")
    conversation_context["current_tag"] = None 
    
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            print("Bot: Take care of yourself!")
            break
        response = get_response(message, intents)
        print("Bot:", response)