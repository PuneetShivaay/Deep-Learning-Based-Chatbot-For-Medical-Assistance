# Medical-Chatbot
It is Deep Learning based chatbot model used for medical assistance. The project is based on a Chatbot for diagnosis of diseases. All diseases have a set of associated symptoms. The patient needs to enter the observed symptoms and the Chatbot can recognize the disease. Whenever someone has some disease the human body responds to it by giving symptoms. These symptoms can point towards a particular disease.

The system is works on the principle of artificial neural networks which simulate human thinking and reasoning. These networks work like the neurons in our brain and simulate medical reasoning. 
 
The input nodes are the set of symptoms and the output nodes are the diseases as recognized by the system based on the set of symptoms. The system gives a value to the diseases and calculates the total a score to all the symptoms and gives a ranking to all the diseases and selects the best ranking disease based on the set of symptoms.


# Requirements
python -v 3.7

pip install -r requirements.txt 


# Instructions
We use mainly these two files :-

1. chatbot_train.py

2. chatbot_gui.py

run python chatbot_train.py

run python chatbot_gui.py


# Details
chatbot_train.py is a python file in which we train the model with the help of available dataset.
Dataset is stored in the json file (intents.json).
chatbot_gui.py is a file which will open a GUI prompt where user can talk with chatbot and interact with it.
