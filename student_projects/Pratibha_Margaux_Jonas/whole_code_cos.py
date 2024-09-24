# -*- coding: utf-8 -*-
"""
Created in the 2024 Computational Linguistics Fall School / Passau
        in the Course 'Python for Language Processing' by Jakob Prange

@authors:
        Pratibha Dongare (pratibhaphdlandp22@efluniversity.ac.in)
        Margaux Susman (Margaux.Susman@uib.no)
        Jonas Lüttke (jonas.luettke@romanistik.uni-freiburg.de)
"""

# ===IMPORT MODULES===
import pathlib
import re
import string
import numpy as np


# ===OPEN RELEVANT FILES===
trainfile = pathlib.Path('en_gum-ud-train.conllu') #open train corpus
stopwords = pathlib.Path('stopwords.txt') #open stopwords
testfile = pathlib.Path('en_gum-ud-test/test_acad_1.txt') #open test corpus (one text extracted from en_gum-ud-test.conllu)

"""
Run the above alternatively with:
    testfile = pathlib.Path('en_gum-ud-test/test_acad_2.txt')
    testfile = pathlib.Path('en_gum-ud-test/test_fict_1.txt')
    testfile = pathlib.Path('en_gum-ud-test/test_fict_2.txt')
    testfile = pathlib.Path('en_gum-ud-test/test_news_1.txt')
    testfile = pathlib.Path('en_gum-ud-test/test_news_2.txt')
    testfile = pathlib.Path('en_gum-ud-test/test_spch_1.txt')
    testfile = pathlib.Path('en_gum-ud-test/test_spch_2.txt')
    testfile = pathlib.Path('en_gum-ud-test/test_whow_1.txt')
    testfile = pathlib.Path('en_gum-ud-test/test_whow_2.txt')
… or with any other text in *.conllu-format.
"""


# ===CREATE VARIABLES===
acadDict = {}
fictDict = {}
newsDict = {}
spchDict = {}
whowDict = {}

currentDict = None # placeholder for the dictionary that is currently used


# ===DEFINE RELEVANT FUNCTIONS===

def remove_punctuation(input_string):
    # Make a translation table that maps all punctuation characters to None
    translator = str.maketrans("", "", string.punctuation)
    # Apply the translation table to the input string
    result = input_string.translate(translator)
    return result

def setDict(genre):
    """set the dictionary currently used to the one of the current genre"""
    if genre == "academic":
        return acadDict
    elif genre == "fiction":
        return fictDict
    elif genre == "news":
        return newsDict
    elif genre == "speech":
        return spchDict
    elif genre == "whow":
        return whowDict


# ===CREATE DICTIONARIES FOR GENRES===
with open(trainfile, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('# meta::genre'): #do this if the line contains genre information
            line = line.split('=')
            line = line[1].strip()
            currentDict = setDict(line)
        elif currentDict != None and re.match(r'\d+', line): #do this if the line is a token from a genre of interest
            line = line.split('\t')
            line = line[1].strip()
            line = line.lower()
            line = remove_punctuation(line)
            if line in currentDict:
                currentDict[line] +=1 #update existing entry
            else:
                currentDict[line] = 1 #create new entry
        else: #do this if the line is not containing genre information and not a token from a genre of interest
            continue

"""""
    #below comes printing for checking
    print('acadDict\n');print(acadDict)
    print('\n\nfictDict\n');print(fictDict)
    print('\n\nnewsDict\n');print(newsDict)
    print('\n\nspchDict\n');print(spchDict)
    print('\n\nwhowDict\n');print(whowDict)
"""""


# ===GROUP DICTIONARIES IN A LIST===
dictionaries = [acadDict, fictDict, newsDict, spchDict, whowDict]
    

# ===CREATE DICTIONARY FROM TEST DATA===
testDict = {}

with open(testfile, encoding='utf-8') as t:
    for line in t:
        line = line.strip()
        if re.match(r'\d+', line): #do this if the line is a token
            line = line.split('\t')
            line = line[1].strip()
            line = line.lower()
            line = remove_punctuation(line)
            if line in testDict:
                testDict[line] += 1
            else:
                testDict[line] = 1
    #print(testDict) #for code debugging only


# ===REMOVE STOPWORDS AND EMPTY STRINGS===
with open(stopwords, encoding='utf-8') as c:
    for stopword in c:
        stopword = stopword.strip()
        for dictionary in dictionaries: # for trained dictionaries from here
            if stopword in dictionary:
                del dictionary[stopword]
            if '' in dictionary:
                del dictionary['']
        if stopword in testDict: # for test dictionary from here
            del testDict[stopword]
        if '' in testDict:
            del testDict['']


# ===MATcH DICTIONARIES===        
def alignDict(baseDict):
    """This function makes sure all dictionaries conatin the same items."""
    for dictionary in dictionaries:
        for word in dictionary:
            if word in baseDict:
                continue
            else:
                baseDict[word] = 0
    for word in testDict:
        if word in baseDict:
            continue
        else:
            baseDict[word] = 0
                
print('Length of dictionaries (need to be identical):')
for dictionary in dictionaries: # for trained dictionaries
    alignDict(dictionary)
    print(len(dictionary)) # print length of dictionaries for debugging purposes

alignDict(testDict)
print(len(testDict))


# ===SORT AND CONVERT DICTIONARIES TO LISTS===
def convDict(dictionary):
    """This function converts the dictionaries to comparable lists by
         (1) sorting them in alphabetical order,
         (2) getting rid of number entries, and
         (3) converting the dictionary into a list.
       We might, eventually, consider splitting it into several functions with each just one task.
       Eventually.
    """
    sortedDict = {key: value for key, value in sorted(dictionary.items())} #sort alphabetically
    filtered_dict = {key: value for key, value in sortedDict.items() if not re.search(r'\d', key)}
    sortedList = list(filtered_dict.values()) #turns dictionary into list
    return sortedList

for i in range(len(dictionaries)):
    dictionaries[i] = convDict(dictionaries[i])
testDict = convDict(testDict)



# ===CONVERT LISTS TO ARRAYS===
for list in dictionaries:
    list = np.array(list)
testDict = np.array(testDict)

"""
#print results to seperate file for processing in Excel (relevant if output doesn't work only')
resultfile = pathlib.Path('results.txt') #open result file
with open(resultfile,'w') as r:
    for array in dictionaries:
        r.write(str(array))
    r.write(str(testArray))
"""

def calcCos(testArray,trainedArray):
    """This function calculates the cosine similarity between the vector representing the test text and one of the vectors representing a genre. It yields a value between -1 (negative correlation) and +1 (absolute positive correlation)."""
    # Calculate dot product
    dotProduct = np.dot(testArray,trainedArray)
    
    # Calculate norms (magnitudes)
    normTest = np.linalg.norm(testArray)
    normTrained = np.linalg.norm(trainedArray)
    
    # Compute cosine similarity
    similarity = dotProduct / (normTest * normTrained)
    
    return similarity

print('\n\nFINDINGS:')
outputTitle = 'genre | score'
outputLine = ''
for i in range(len(outputTitle)):
    outputLine += '='
print('\n'+outputTitle+'\n'+outputLine)

arrayNames = ['acad','fict','news','spch','whow'] #amend if dictionaries used change!!
i = 0

for array in dictionaries:
    cosRes = calcCos(testDict,array)
    cosRes = round(cosRes,3)
    print(arrayNames[i],' |',cosRes)
    i += 1
print("""\n^ The scores indicate the lexical cosine similarity between the test text and an average text from the according genres.
+1 indicates a 100% similarity.
 0 indicates no similarity.
-1 indicates negative similarity.""")