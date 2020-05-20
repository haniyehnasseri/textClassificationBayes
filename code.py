import csv
import re
import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer
import pandas as pd
import random
import datetime

import nltk
from nltk.corpus import wordnet
#nltk.download('wordnet')
#nltk.download('punkt')

count = 0
description_Travel = []
description_Business = []
description_Beauty = []


lemmatizer = WordNetLemmatizer()
def nltk2tag(tag):
  if tag.startswith('J'):
    return wordnet.ADJ
  elif tag.startswith('V'):
    return wordnet.VERB
  elif tag.startswith('N'):
    return wordnet.NOUN
  elif tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None
def lemmatize(text):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    tagged = map(lambda x: (x[0], nltk2tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return res_words

#def testOutput():
    #all = 0
    #correct = 0
    #pd1 = pd.read_csv('output.csv')
    #pd2 = pd.read_csv('ans.csv')
    #joined = pd.merge(pd1,pd2,on='index')
    #for ind in joined.index:
        #if(joined['category_x'][ind] != ''):
            #all += 1
            #if(joined['category_x'][ind] == joined['category_y'][ind]):
                #correct += 1

    #print("Output accuracy Result: ")
    #print(correct/all)
    



def normalize(text):
    normalizedText = ""
    normalizedText = text.lower()
    normalizedText = re.sub(r'\d+', '', text)
    for p in string.punctuation:
        normalizedText = normalizedText.replace(p, ' ')
    normalizedText = re.sub(' +', ' ', normalizedText)
    normalizedText = normalizedText.strip()

    stop_words = set(stopwords.words('english'))

    stemmedNormalizedTextList = lemmatize(normalizedText)
    stopWordFreeNormalizedTextList = []

    for w in stemmedNormalizedTextList:
        if w not in stop_words:
            stopWordFreeNormalizedTextList.append(w)
    return stopWordFreeNormalizedTextList



def testProcess(testFileName,ClassNames,vocabularyDictionary,classTowordDict,totalWordDict,learn):

    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        allNews = 0
        for subj in learn:
            allNews += len(learn[subj])
    
        with open(testFileName) as csvfile:
            readCSV = csv.reader(csvfile)
            count = 0
            for row in readCSV:
                if(count == 0):
                    count += 1
                    writer.writerow(["index", "category"])
                    continue
                
                if(row[4] == ''):
                    writer.writerow([count - 1, ''])
                    count += 1
                    continue

                news = normalize(row[4])
                
                _max = -1
                chosenClass = ''
                for c in ClassNames:
                    mult = 1
                    for word in news:
                        if(word not in classTowordDict[c]):
                            classTowordDict[c][word] = 0
                        mult *= (classTowordDict[c][word] + 1)/(totalWordDict[c] + len(vocabularyDictionary[c]) * 1)

                    if(mult * (len(learn[c])/allNews) > _max):
                        _max = mult * (len(learn[c])/allNews)
                        chosenClass = c

                writer.writerow([count - 1, chosenClass])
                count += 1
        
        
        
        
def textProcess(classNames,Classes,learn,train,vocabularyDictionary,classTowordDict,totalWordDict):
        
    for i in range(0,len(Classes)):
        classSize = len(Classes[i])
        learn[classNames[i]] = []
        train[classNames[i]] = []
        for j in range(0,int(0.80 * classSize)):
            learn[classNames[i]].append(normalize(Classes[i][j][1] + " " + Classes[i][j][4] + " " + Classes[i][j][6]))

        for j in range(int(0.80 * classSize),int(classSize)):
            train[classNames[i]].append(normalize(Classes[i][j][6]))
            

    for c in classNames:
        classTowordDict[c] = dict()
        totalWordDict[c] = 0
        vocabularyDictionary[c] = set()
        
    for subj in learn:
        
        for news in learn[subj]:
            for word in news:
                vocabularyDictionary[subj].add(word)
                totalWordDict[subj] += 1
                if(word not in classTowordDict[subj]):
                    classTowordDict[subj][word] = 1
                else:
                    classTowordDict[subj][word] += 1

                    
    
    
    
    predictedClasses = dict()
    corrects = dict()
    allNews = 0
    for subj in learn:
        allNews += len(learn[subj])
        predictedClasses[subj] = 0
        corrects[subj] = 0

    recalls = dict()
    precisions = dict()
    allCorrects = 0
    confusion_matrix = dict()
    
    
        
    for subj in classNames:
        
        confusion_matrix[subj] = dict()
        for news in train[subj]:
            _max = -1
            chosenClass = ''
            for c in classNames:
                mult = 1
                for word in news:
                    if(word not in classTowordDict[c]):
                        classTowordDict[c][word] = 0
                    mult *= (classTowordDict[c][word] + 1)/(totalWordDict[c] + len(vocabularyDictionary[c]) * 1)

                if(mult * (len(learn[c])/allNews) > _max):
                    _max = mult * (len(learn[c])/allNews)
                    chosenClass = c
            

            predictedClasses[chosenClass] += 1
            
            if(chosenClass not in confusion_matrix[subj]):
                confusion_matrix[subj][chosenClass] = 1
            else:
                confusion_matrix[subj][chosenClass] += 1
                
            if(subj == chosenClass):
                corrects[subj] += 1

        allCorrects += corrects[subj]


    allTrainedNews = 0
    for subj in train:
        allTrainedNews += len(train[subj])
        recalls[subj] = corrects[subj]/len(train[subj])
        precisions[subj] = corrects[subj]/predictedClasses[subj]
        confusion_matrix[subj]['Total'] = len(train[subj])

    print("Recalls :",recalls)
    print("Precisions :",precisions)
    print("Accuracy :",allCorrects/allTrainedNews)
    print("confusion_matrix :",confusion_matrix)



def sampling(description_Beauty,description_Business):
    for i in range(0,len(description_Travel) - len(description_Business)):
        description_Business.append(random.choice(description_Business))


    for i in range(0,len(description_Travel) - len(description_Beauty)):
        description_Beauty.append(random.choice(description_Beauty))





    random.shuffle(description_Business)
    random.shuffle(description_Business)
    random.shuffle(description_Beauty)
    random.shuffle(description_Beauty)
    
    

with open('data.csv') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV:
        if(count == 0):
            count += 1
            continue
        if(row[2] == 'STYLE & BEAUTY'):
            description_Beauty.append(row)
        if(row[2] == 'BUSINESS'):
            description_Business.append(row)
        if(row[2] == 'TRAVEL'):
            description_Travel.append(row)





sampling(description_Beauty,description_Business)

learn = dict()
train = dict()
vocabularyDictionary = dict()
classTowordDict = dict()
totalWordDict = dict()

start = datetime.datetime.now()
textProcess(['BUSINESS','TRAVEL'],[description_Business,description_Travel],learn,train,vocabularyDictionary,classTowordDict,totalWordDict)


learn = dict()
train = dict()
vocabularyDictionary = dict()
classTowordDict = dict()
totalWordDict = dict()

textProcess(['BUSINESS','TRAVEL','STYLE & BEAUTY'],[description_Business,description_Travel,description_Beauty],learn,train,vocabularyDictionary,classTowordDict,totalWordDict)

testFileName = 'test.csv'
testProcess(testFileName,['BUSINESS','TRAVEL','STYLE & BEAUTY'],vocabularyDictionary,classTowordDict,totalWordDict,learn)
finish = datetime.datetime.now()
#testOutput()
print(finish - start)


