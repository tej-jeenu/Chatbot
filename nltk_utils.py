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
    genpositivesentiment = ["yes","yeah","yep","regularly","all the time","often","daily","do it","too much","I do"]
    genmediumsentiment = ["sometimes","inconsistent","weekly","every now and then","inconsistently","try to"]
    gennegativesentiment = ["no","don't","never done it before","never did it before","not","never","rarely","rare"]

    genpositivesentiment.extend(specpositivesentiment)
    genmediumsentiment.extend(specmediumsentiment)
    gennegativesentiment.extend(specnegativesentiment)

    if any(word in text for word in genmediumsentiment):
        sentiment = "medium"
    elif any(word in text for word in gennegativesentiment):
        sentiment = "negative"
    elif any(word in text for word in genpositivesentiment):
        sentiment = "positive"
    
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
    sentencelist = inputtext.split(",")
    sleep = int(sentencelist[len(sentencelist)-1])
    sentencelist.remove(sentencelist[len(sentencelist)-1])
    for sentence in sentencelist:
        newtag,output,predicted = predicttag(sentence)
        taglist.append(newtag)
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



def followupconvo():
    global intents
    global taglist
    global tagcounter
    global followupcounter
    global currentq
    global finished
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
        if(sleep < 8):
            sleep=8
            todolist.append("SLEEP for 8 HOURS")
        for item in todolist:
            print("-" + item)
            finished = True


def submitcmd(inputtext):
    chattextbox = ""
    inputtextbox = inputtext
    response = ""
    global currentq
    global initialstate
    if(initialstate == True):
        convostart(inputtextbox)
        followupconvo()
        chattextbox = currentq
        initialstate = False
    else:
        response = processresponse(inputtextbox)
        followupconvo()
        if(finished == False):
            chattextbox = response + "\n \n" + currentq
    
    return chattextbox


#while(finished==False):
#    inputtext = input("respond to chatbot:")
#    chattext = submitcmd(inputtext)
#    print(chattext)






    













