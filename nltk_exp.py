import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
#from collections import Counter

import pickle

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        
        votes=[]
        for c in self._classifiers:
            v= c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf
        
        

documents=[]

for cat in movie_reviews.categories():
    for ids in movie_reviews.fileids(cat):
        documents.append([list(movie_reviews.words(ids)),cat])

random.shuffle(documents)

all_words = []
for word in movie_reviews.words():
    all_words.append(word.lower())

all_words=nltk.FreqDist(all_words)
all_words=sorted(all_words,key=all_words.get,reverse=True)


word_features=all_words[:3000]

def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]= (w in words)
    return features
    
#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

feature_sets=[(find_features(rev),category) for (rev,category) in documents]

training_set = feature_sets[:1500]
testing_set = feature_sets[1500:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print(" Original Naive Bayes Accuracy:",nltk.classify.accuracy(classifier,testing_set)*100)

#save_classifier = open("naivebayes.pkl","wb")
#pickle.dump(classifier , save_classifier)
#save_classifier.close()
#classifier.show_most_informative_features(15)


mnb_classifier = SklearnClassifier(MultinomialNB())
mnb_classifier.train(training_set)
print(" MultiNomial Naive Bayes Accuracy:",nltk.classify.accuracy(mnb_classifier,testing_set)*100)

#gnb_classifier = SklearnClassifier(GaussianNB())
#gnb_classifier.train(training_set)
#print(" Gaussian Naive Bayes Accuracy:",nltk.classify.accuracy(gnb_classifier,testing_set)*100)

bnb_classifier = SklearnClassifier(BernoulliNB())
bnb_classifier.train(training_set)
print(" Bernoulli Naive Bayes Accuracy:",nltk.classify.accuracy(bnb_classifier,testing_set)*100)


#LogisticRegression, SGDClassifier
#SVC, LinearSVC, NuSVC
#LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
#LogisticRegression_classifier.train(training_set)
#print(" LogisticRegression Accuracy:",nltk.classify.accuracy(LogisticRegression_classifier,testing_set)*100)

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print(" SGDClassifier Accuracy:",nltk.classify.accuracy(SGD_classifier,testing_set)*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print(" SVC Accuracy:",nltk.classify.accuracy(SVC_classifier,testing_set)*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print(" LinearSVC Accuracy:",nltk.classify.accuracy(LinearSVC_classifier,testing_set)*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print(" NuSVC Accuracy:",nltk.classify.accuracy(NuSVC_classifier,testing_set)*100)

    
voted_classifier = VoteClassifier(classifier,
                                  mnb_classifier,
                                  bnb_classifier,
                                  SGD_classifier,
                                  SVC_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print(" voted_classifier Accuracy:",nltk.classify.accuracy(voted_classifier,testing_set)*100)

print('Classification:',voted_classifier.classify(testing_set[0][0]),'Confidence:',voted_classifier.confidence(testing_set[0][0]))

print('Classification:',voted_classifier.classify(testing_set[5][0]),'Confidence:',voted_classifier.confidence(testing_set[5][0]))

