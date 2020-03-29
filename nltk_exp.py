import nltk
import random
from nltk.corpus import movie_reviews
import pickle

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
print("Naive Bayes Accuracy:",nltk.classify.accuracy(classifier,testing_set)*100)

save_classifier = open("naivebayes.pkl","wb")
pickle.dump(classifier , save_classifier)
save_classifier.close()
#classifier.show_most_informative_features(15)



    

