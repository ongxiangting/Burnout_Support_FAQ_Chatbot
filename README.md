# Burnout_Support_FAQ_Chatbot

**Group Number:** Group [3]
**Programme:** Bachelor in Data Science
**Group Members:**
1. Ong Xiang Ting (26WMR12525) - Machine Learning Approach (Feedforward Neural Network)
2. Lai Zhen Xuan (26WMR12499) - Platform-Based Approach (Google Dialogflow)

## 1. Project Purpose
This project implements an Artificial Intelligence chatbot designed to provide conversational support for occupational and academic burnout. The system classifies user intents ranging from burnout symptoms to crisis support, offering safe and contextually appropriate responses.

## 2. Main Prototype Functions
* **Intent Classification:** Accurately maps user text to over 43 distinct burnout-related intents.
* **Confidence Threshold & Fallback:** Routes ambiguous or unrecognized queries to a safe fallback response to prevent blind guessing.
* **Dual Approach Comparison:** Features a custom-built Feedforward Neural Network (FNN) and a Google Dialogflow implementation for comparative analysis.

## 3. Technology Stack & Execution Environment
* **Programming Language:** Python 3.9+
* **Frameworks/Libraries:** TensorFlow/Keras, NLTK, Scikit-learn, NumPy, Matplotlib
* **Supported OS:** Windows 10/11, macOS, Linux

## 4. Installation Instructions
To evaluate the prototype, please extract the ZIP file and follow these exact steps in your terminal/command prompt:

1. Navigate to the project root folder:
   `cd Group3_BurnoutChatbot`
2. Create and activate a virtual environment (Recommended):
   `python -m venv venv`
   *Windows:* `venv\Scripts\activate`
   *Mac/Linux:* `source venv/bin/activate`
3. Install all required dependencies:
   `pip install -r Installation_and_User_Guide/requirements.txt`
4. (First-time NLTK setup): The script will automatically download required NLTK packages (`punkt`, `wordnet`) upon running.

## 5. Dataset and Trained Model Setup
* **Dataset:** Located in `Dataset/intents.json`. No further setup is required as the script reads directly from this file.
* **Trained Model:** The trained FNN model (`chatbot_model.h5`), word vocabulary (`words.pkl`), and classes list (`classes.pkl`) are pre-saved in the `Trained_Model/` folder. The program will load these automatically. You do **not** need to retrain the model to test the prototype.

## 6. How to Run the Prototype
To start the Machine Learning chatbot interface, run the following command from the root directory:
`python Source_Code/main.py`
*(Wait a few seconds for TensorFlow to load the model. The terminal will display "This is Burnout Support Chatbot!" when you can start typing).*

## 7. Test Inputs and Expected Outputs
Here are sample inputs to test the core functions:
* **Normal/Success Case:** 
  * *Input:* "I am feeling so exhausted from studying."
  * *Expected Output:* The bot will recognise the `student_burnout` intent and explain that students can experience burnout due to academic pressure.
* **Different Scenario:**
  * *Input:* "How do I set boundaries at work?"
  * *Expected Output:* The bot will recognise the `boundaries_detail` intent and provide tips on separating work and personal life.
* **Invalid/Fallback Case (Threshold Test):**
  * *Input:* "asdfghjkl" (or an entirely unrelated topic like "What is the capital of France?")
  * *Expected Output:* The model's confidence will fall below the threshold, triggering the fallback response: "I'm sorry, I didn't quite understand that. Could you rephrase?"

## 8. Known Limitations & Troubleshooting
* **Context limitation:** The Bag-of-Words (BoW) model ignores word sequence. Extremely complex or compound sentences might trigger the fallback response.
* **Troubleshooting `ModuleNotFoundError`:** Ensure the virtual environment is activated before running the `pip install` command.
* **Troubleshooting NLTK errors:** If NLTK fails to download data automatically due to firewall restrictions, run python and type `import nltk; nltk.download('all')` manually.

## 9. Group Member Mapping and Contribution
* **Ong Xiang Ting:** Responsible for the Custom Machine Learning Solution. Implemented the text preprocessing pipeline (NLTK), the Bag-of-Words vectorizer, the Feedforward Neural Network architecture (TensorFlow/Keras), model training/evaluation, and the `main.py` terminal interface.
* **Lai Zhen Xuan:** Responsible for the Platform-Based Solution. Designed and configured the Google Dialogflow agent, defined intent training phrases, managed conversational flow, and integrated the platform response testing.
