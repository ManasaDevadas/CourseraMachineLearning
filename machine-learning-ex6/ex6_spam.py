from itertools import accumulate
import scipy.io
import numpy as np
from sklearn import svm
import re
import string
from nltk.stem import PorterStemmer

ps = PorterStemmer()

#steps done - 1. create a vocab dictionary datatype from the word dict given. 2. a. process email, do stemming 
# 2b. match and find indices of the words appearing in the email. 2c. create a feature vector with only ones and zeros, ones at 
#position where word in dict appears, 0 otherwise. 
#3. spamTrain.mat contains multiple feature vectors for emails and corresponding y values., train using that
#4. Predict and measure accurary with the similar test dataset.
#5. Top predictors, model.coef would be weights of featureVector which inturn corresponds to the position in dictionary, 
# use numpy argsort to understand the index of highest weights. Use the index to find corresponding word in vocab_dict.
#6. In train and test we used readymade featurevector, now use process email and email features to create a feature vector 
# from a sample email and predict using the model created.

#The vocabulary list is a text file with no and word, convert that to Dict
def get_vocab_list():
    vocab_dict = {}
    with open('vocab.txt') as f:
        for line in f:
                (val, key) = line.split()
                #vocab_dict[int(val)] = key
                vocab_dict[key] = val
    print("length of vocab dict:", len(vocab_dict))
    #position = val_list.index(1190)
    return vocab_dict

def processEmail(email_contents):
        
        word_indices = []
        vocab_dict = get_vocab_list()

        #Preprocess email
        email_contents = email_contents.lower()

        #replace all the given <[^<>]+> with space
        email_contents = re.sub('<[^<>]+>', ' ', email_contents)

         # Any numbers get replaced with the string 'number'
        email_contents = re.sub('[0-9]+', 'number', email_contents)

         # The '$' sign gets replaced with 'dollar'
        email_contents = re.sub('[$]+', 'dollar', email_contents)

        # URLs
        email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)
            # ===================== Tokenize Email =====================

         # Output the email
        print('==== Processed Email ====')
        print(email_contents)
        
        words = re.split('[@$/#.-:&*+=\[\]?!(){\},\'\">_<;% ]', email_contents)

        for word in words:
                word = re.sub('[^a-zA-Z0-9]', '', word)
                #Do stemming of the word
                word = ps.stem(word)
                #print("after stemming")
                #print("word", word)

                if len(word) < 1:
                        continue
        # ===================== Your Code Here =====================
        # Instructions : Fill in this function to add the index of token to
        #                word_indices if it is in the vocabulary. At this point
        #                of the code, you have a stemmed word frome email in
        #                the variable token. You should look up token in the
        #                vocab_list. If a match exists, you should add the
        #                index of the word to the word_indices nparray.
        #                Concretely, if token == 'action', then you should
        #                look up the vocabulary list the find where in vocab_list
        #                'action' appears. For example, if vocab_list[18] == 'action'
        #                then you should add 18 to the word_indices array.



        # ==========================================================
                if word in vocab_dict:
                        #print ("word in vocab", word)
                        word_indices.append(int(vocab_dict[word]))
        

        #print("word Indices:\n", word_indices)
        return word_indices

def email_features(word_indices):
        #EMAILFEATURES takes in a word_indices vector and produces a feature vector
        #from the word indices
        #   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
        #   produces a feature vector from the word indices. 

        # Total number of words in the dictionary
        n = 1899
        # Feature vector
        # index of numpy array starts at 0, but the word indices starts at 1.
        featurevector = np.zeros([n, 1])
        for index in word_indices:
                featurevector[index - 1] = 1 # -1 because python lists start at 0
        
        return featurevector



        



    
def main() : 

        file = open('emailSample1.txt')
        file_contents = file.read()
        file.close()
        print(file_contents)

        #process the email
        word_indices = processEmail(file_contents)
        print('Expected output:')
        print('86 916 794 1077 883 370 1699 790 1822 1831 ...\n')
        print('word_indices:')
        print(word_indices[:10])

        # ===================== Part 2: Feature Extraction =====================
        # Now, you will convert each email into a vector of features in R^n.
        # You should complete the code in emailFeatures.py to produce a feature
        # vector for a given mail

        print('Extracting Features from sample email (emailSample1.txt) ... ')

        # Extract features
        features = email_features(word_indices)

        # Print stats
        print('Length of feature vector: {}'.format(features.size))
        print('Number of non-zero entries: {}'.format(np.flatnonzero(features).size))

        #input('Program paused. Press ENTER to continue')

        # Training SVM for Spam Classification
        emails_train = scipy.io.loadmat('spamTrain.mat')
        print(emails_train.keys())
        print("shapeof X", emails_train['X'].shape)
        # Train the SVM
        X = emails_train['X']
        y = emails_train['y']
        y = y.flatten()

        model = svm.SVC(kernel='linear', C=0.1)
        model.fit(X, y)
        pred = model.predict(X)
        print('Training accuracy:', np.mean(pred == y) * 100)

        #Evaluate on test set (this is a featureset, not emails)
        emails_test  = scipy.io.loadmat('spamTest.mat')
        print(emails_test.keys())
        X = emails_test['Xtest']
        y = emails_test['ytest']
        y = y.flatten()
        pred= model.predict(X)
        print("accuracy on test set:",  np.mean(pred == y) * 100)

        #================= Part 5: Top Predictors of Spam ====================
        #  Since the model we are training is a linear SVM, we can inspect the
        #  weights learned by the model to understand better how it is determining
        #  whether an email is spam or not. The following code finds the words with
        #  the highest weights in the classifier. Informally, the classifier
        #  'thinks' that these words are the most likely indicators of spam.

       # https://stackoverflow.com/questions/26984414/efficiently-sorting-a-numpy-array-in-descending-order
       #https://numpy.org/doc/stable/reference/generated/numpy.argsort.html 

        vocab_dict = get_vocab_list()
        #reverse key and value to have no as key and word as value in index_dict
        index_dict = {v: k for k, v in vocab_dict.items()}
        print("length of index dict:", len(index_dict))
        print((model.coef_).shape)
        print((model.coef_))
        #argsort - Returns the indices that would sort an array.
        indices = np.argsort(model.coef_).flatten()[::-1]



        for i in range(15):
                #print(i)
                print(indices[i])
                print(index_dict[str(indices[i])], model.coef_.flatten()[indices[i]])

        #Optional exercise: Predict some emails
        email_files = ['emailSample1.txt', 'emailSample2.txt', 'spamSample1.txt', 'spamSample2.txt']

        for email_file in email_files:
                with open(email_file) as f:
                        email_contents = f.read()
                word_indices = processEmail(email_contents)
                featurevector = email_features(word_indices)
                #print("shape of featurevector")
                #print(np.shape(featurevector))
                #print(featurevector)
                #Take transpose to match the dimension of X used to get trained model.
                featurevector = featurevector.T
                pred = model.predict(featurevector)  
                print(email_file, pred)  
                


if __name__ == "__main__":
        main()