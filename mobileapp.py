from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.list import MDList, OneLineAvatarListItem, IconLeftWidget, IconRightWidget
from kivymd.uix.button import MDIconButton,MDFillRoundFlatButton
from kivymd.uix.card import MDCard
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.selectioncontrol import MDCheckbox

import numpy as np
import json
import torch
import torch.nn as nn
import random
import time
import spacy
# load english language model and create nlp object from it
nlp = spacy.load("en_core_web_sm")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

taglist = []
todolist = []
sleep = 0
initialstate = True
currentq = ""
tagcounter = 0
followupcounter = 0
finished = False

genpositivesentiment = ["yes","yeah","yep","regularly","all the time","often","daily","do it","too much","I do"]
genmediumsentiment = ["sometimes","inconsistent","weekly","every now and then","inconsistently","try to"]
gennegativesentiment = ["no","don't","never done it before","never did it before","not","never","rarely","rare"]


with open('chatbotqs.json', 'r') as json_data:
    chatbotqs = json.load(json_data)

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    #return nltk.word_tokenize(sentence)
    doc = nlp(sentence)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return filtered_tokens


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """

     
    #return stemmer.stem(word.lower())
    return word


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


def predicttag(inputtext:str):
    FILE = "testmodel.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    sentence = tokenize(inputtext)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    return tag,output,predicted
#method workd correctly

def predictsentiment(text:str,specpositivesentiment:list,specmediumsentiment:list,specnegativesentiment:list):
    
    if any(word in text for word in specmediumsentiment):
        sentiment = "medium"
    elif any(word in text for word in specnegativesentiment):
        sentiment = "negative"
    elif any(word in text for word in specpositivesentiment):
        sentiment = "positive"
    else:
        sentiment = "negative"
    
    return sentiment
#method works correctly


def getusualresponse(inputtext:str):
    global intents
    response = ""
    tag,output,predicted = predicttag(inputtext)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
    else:
        response = "Sorry I do not understand!"

    return response
#method works correctly

def convostart(inputtext:str):
    global taglist
    global sleep
    sentencelist = inputtext.split("\n-")
    sleep = int(sentencelist[len(sentencelist)-1])
    sentencelist.remove(sentencelist[len(sentencelist)-1])
    for sentence in sentencelist:
        newtag,output,predicted = predicttag(sentence)
        taglist.append(newtag)
    
    taglist = list(dict.fromkeys(taglist))
    
    if("Mental-Health" in taglist):
        if("Purpose" in taglist):
            taglist.remove("Purpose")
        if("Intelligence" in taglist):
            taglist.remove("Intelligence")

    elif("Appearance" in taglist):
        if("Skin" in taglist):
            taglist.remove("Skin")
        if("Body" in taglist):
            taglist.remove("Body")

    
#method works correctly

def converttominute(target:float, lemma:str):
    if(lemma == "hour"):
        target = target*60
    if(lemma == "second"):
        target = target/60
    return target
#this method works correctly


def numericanalysis(text:str):
    global chatbotqs
    global currentq
    global todolist

    numericq = False
    numericval = 0
    sentiment = ""
    sentencelist = tokenize(text)
    for lemma in sentencelist:
        lemma = str(lemma)
        if(lemma.isnumeric()):
            numericval = float(lemma)
            numericq = True
    
    if(numericq == False):
        itag = predictsentiment(text,chatbotqs[currentq]["positive"]["patterns"],chatbotqs[currentq]["medium"]["patterns"],chatbotqs[currentq]["negative"]["patterns"])
        sentiment = str(itag)
    elif(numericq == True):
        for lemma in sentencelist:
            if(lemma == "hour"):
                numericval = converttominute(numericval, "hour")
            elif(lemma == "second"):
                numericval = converttominute(numericval, "second")
        if(numericval >= float(chatbotqs[currentq]["positive"]["bound"])):
            sentiment = "positive"
        elif(numericval >= float(chatbotqs[currentq]["medium"]["bound"])):
            sentiment = "medium"
        elif(numericval >= float(chatbotqs[currentq]["negative"]["bound"])):
            sentiment = "negative"
    
    response = random.choice(chatbotqs[currentq][sentiment]["responses"])
    response = str(response)
    if(sentiment == "negative" or sentiment == "medium"):
        todolist.extend(chatbotqs[currentq][sentiment]["actions"])
    
    return response
#this method works correctly
           
def textanalysis(text:str):
    global currentq
    global todolist
    global chatbotqs

    sentiment = predictsentiment(text,chatbotqs[currentq]["positive"]["patterns"],chatbotqs[currentq]["medium"]["patterns"],chatbotqs[currentq]["negative"]["patterns"])
    
    response = random.choice(chatbotqs[currentq][sentiment]["responses"])
    if(sentiment == "negative" or sentiment == "medium"):
        todolist.extend(chatbotqs[currentq][sentiment]["actions"])
    
    return response
#this method works correctly


def processresponse(text:str):
    global chatbotqs
    global currentq
    response = ""
    if(chatbotqs[currentq]["type"] == "numeric"):
        response = numericanalysis(text)# text implying the input text
    if(chatbotqs[currentq]["type"] == "text"):
        response = textanalysis(text)
    
    return response
#this method works correctly

class SelfImprovementChatbot(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Light"  # Choose the desired theme style

        def followupconvo():
            global intents
            global taglist
            global tagcounter
            global followupcounter
            global finished
            global currentq
            global sleep
            global todolist

            
            if(tagcounter < len(taglist)):
                qtag = taglist[tagcounter]
                for intent in intents['intents']:
                    if qtag == intent["tag"]:
                        currentq = intent["followup"][followupcounter]
                        if(followupcounter == len(intent["followup"])-1):
                            followupcounter = 0
                            tagcounter = tagcounter + 1
                        else:
                            followupcounter = followupcounter + 1
            else:
                todolist = list(dict.fromkeys(todolist))
                if(sleep < 8):
                    sleep=8
                    todolist.append("shut down and put away phone in a different room to where you sleep")
                    todolist.append("sleep 8 hours minimum")
                for item in todolist:
                    listicon = IconLeftWidget(icon="chevron-right-box-outline")
                    item = OneLineAvatarListItem(text=item)
                    item.add_widget(listicon)
                    listlayout.add_widget(item)
                finished = True


        def sendquery(instance):
           response = ""
           global currentq
           global initialstate
           global finished
           if(initialstate == True):
                convostart(querysection.text)
                followupconvo()
                answersection.text = currentq
                querysection.text = ""
                initialstate = False
           else:
                response = processresponse(querysection.text)
                followupconvo()
                if(finished == False):
                    answersection.text = response + ", " + currentq
                    querysection.text = ""
                else:
                    answersection.text = "here's a list of tasks to follow, I recommend writing these down on a physical todolist or whiteboard"
                    querysection.text = ""
        
        def resetall(instance):
            global initialstate
            global taglist
            global todolist
            global sleep
            global currentq
            global tagcounter
            global followupcounter
            global finished
            listlayout.clear_widgets()
            taglist = []
            todolist = []
            sleep = 0
            initialstate = True
            currentq = ""
            tagcounter = 0
            followupcounter = 0
            finished = False
            answersection.text = "Hi, please can you bullet point your current problems with dashes before it, aspirations\nand could you also give the number of hours of sleep you usually get in hours at the end?"


        # Create a vertical box layout to arrange the text fields
        box_layout = MDBoxLayout(
            orientation="vertical", 
            spacing="50dp", 
            padding="30dp", 
            pos_hint = {"top":1},
            adaptive_height = True
        )

        # First text field
        querysection = MDTextField(
            multiline = True,
            hint_text="Enter Query/Response Here",
            helper_text_mode="on_error",
            icon_right="human-handsup",
            mode = "fill",
            #icon_right_color=(0, 0, 0, 1),
        )
        box_layout.add_widget(querysection)

        submitquerybtn = MDIconButton(icon = "send", pos_hint = {'center_x':0.5}, on_release = sendquery)
        box_layout.add_widget(submitquerybtn)

        # Second text field
        answersection = MDTextField(
            multiline = True,
            hint_text="Chatbot response",
            helper_text_mode="persistent",
            icon_right="robot",
            readonly = True,
            mode = "fill",
            #icon_right_color=(0, 0, 0, 1),
        )
        box_layout.add_widget(answersection)
        answersection.text = "Hi, please can you bullet point your current problems, aspirations\nand could you also give the number of hours of sleep you usually get in hours at the end?"

        clearbtn = MDIconButton(icon = "reload", pos_hint = {'center_x':0.5}, on_release = resetall)
        box_layout.add_widget(clearbtn)

        listlayout = MDList()
        box_layout.add_widget(listlayout)

        scroll = MDScrollView()
        scroll.add_widget(box_layout)

        return scroll
    
    



if __name__ == "__main__":
    SelfImprovementChatbot().run()