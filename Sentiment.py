# -*- coding: utf-8 -*-
import re
import csv
import nltk
import keras
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from nltk.corpus import stopwords
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential,Model,load_model
from keras.layers import Embedding,Conv1D,MaxPooling1D
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau,EarlyStopping
from keras.applications import Xception
from keras import regularizers
from keras import backend as K

seed=5
np.random.seed(seed)
#loading train data , caluclate Mean & subtract it data, gets the COV. Matrix.(include cleaning the data)
def load_TrainData(path):     
    Data = pd.read_csv(path, sep='\t', header=0)
    X_train = np.array(list(Data['Phrase']))
    Y_train = np.array(list(Data['Sentiment']))
    print(Data)
    print(X_train,Y_train)
    return  X_train, Y_train
#loading test data , caluclate Mean & subtract it data, gets the COV. Matrix.(include cleaning the data)
def load_TestData(path):     
    Data = pd.read_csv(path, sep='\t', header=0)
    X_test=np.array(list(Data['Phrase']))
    X_test_PhraseID=np.array(list(Data['PhraseId']))
    return  X_test,X_test_PhraseID

def shuffle_data(a, b): # Shuffles 2 arrays with the same order
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]

def LSTM_network(train_path,test_path):
    X_train, Y_train = load_TrainData(train_path)
    X_test,X_test_PhraseID = load_TestData(test_path)
    print ('==============================The shape of Training data & Testing data ==============================')
    print("X_train shape is",X_train.shape)
    print("Y_train shape is",Y_train.shape)
    print("X_test shape is",X_test.shape)
    print("X_test_PhraseID shape is",X_test_PhraseID.shape)


    tokenizer = Tokenizer()
    #To create token dictionary, every element is a document(use train and test data to encure the integrity of token dictionary)
    tokenizer.fit_on_texts(np.concatenate((X_train, X_test), axis=0))
    #calculate the word dictionary size exclude the same word 
    tokenizer_vocabulary_size = len(tokenizer.word_index) + 1
    print("Vocabulary size",tokenizer_vocabulary_size)
    # print(type(X_train))
    # print("Word index",Tokenizer.word_index)

    #split the data, use 20% of whole data as validation data 
    Split_number = 31212

    Y_val = Y_train[:Split_number]
    X_val = X_train[:Split_number]

    X_train = X_train[Split_number:]
    Y_train = Y_train[Split_number:]

    #set the max word  and max dictionary size for building LSTM network ( as embedding parameters)
    maxWord= 60
    Dictionary_size=tokenizer_vocabulary_size

    #transform the document to vector shape, the shape type is [the number of document, the length per document]
    encoded_X_train = tokenizer.texts_to_sequences(X_train)
    encoded_X_val = tokenizer.texts_to_sequences(X_val)
    encoded_X_test = tokenizer.texts_to_sequences(X_test)

    #padding all text to same size
    X_train_encoded = sequence.pad_sequences(encoded_X_train, maxlen=maxWord)
    X_val_encoded = sequence.pad_sequences(encoded_X_val, maxlen=maxWord)
    X_test_encoded= sequence.pad_sequences(encoded_X_test, maxlen=maxWord)

    # One Hot Encoding
    Y_train = keras.utils.to_categorical(Y_train, 5)
    Y_val   = keras.utils.to_categorical(Y_val, 5)

    #shuffling the traing Set
    shuffle_data(X_train_encoded,Y_train)

    #Build the LSTM network
    model = Sequential()
    #change words to int type
    model.add(Embedding(Dictionary_size, 32, input_length=maxWord))
    # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.5))
    # model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    #hidden layers
    model.add(LSTM(64))
    # model.add(Flatten())
    model.add(Dropout(0.5))#Freeze some Neuron to avoid overestimate
    model.add(Dense(1200, activation='relu',W_constraint=maxnorm(1)))
    # model.add(Dropout(0.6))
    model.add(Dense(500, activation='relu',W_constraint=maxnorm(1)))

    # model.add(Dropout(0.5))
    #output layer
    model.add(Dense(5, activation='softmax'))

    # Compile model
    model.summary()

    learning_rate=0.001
    epochs = 10
    batch_size = 64 #32
    sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
    Nadam = keras.optimizers.Nadam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/log_1', histogram_freq=0, write_graph=True, write_images=False)#Create the daily record
    checkpointer = ModelCheckpoint(filepath="./weights/weights_1.hdf5", verbose=1, save_best_only=True, monitor="val_loss")#For saving the model between the epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)# control the learning rate
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)#when early stop is actived, after the number of patience epochs stop training

    #Loading best weights
    # model.load_weights("./weights/weights_19.hdf5")

    print ("=============================== Training =========================================")

    # uncommit this to train
    # tensorboard --logdir=./logs

    history  = model.fit(X_train_encoded, Y_train, epochs = epochs, batch_size=batch_size, verbose=1,
                        validation_data=(X_val_encoded, Y_val), callbacks=[tensorboard, reduce_lr,checkpointer,earlyStopping])

    print ("=============================== Score_Accuarcy =========================================")
    scores = model.evaluate(X_val_encoded, Y_val, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print ("=============================== Predicting =========================================")

    f = open('Submission_result.csv', 'w')
    f.write('PhraseId,Sentiment\n')

    # Get the predicted data
    predicted = model.predict_classes(X_test_encoded, batch_size=batch_size, verbose=1)
    for i in range(0,X_test_PhraseID.shape[0]):
        f.write(str(X_test_PhraseID[i])+","+str(predicted[i])+'\n')

    f.close()
    print("================================Done===============================")

    # Count the epoch
    epoch_count = range(1, len(history.history['loss']) + 1)

    #visulize the training process
    plt.plot(epoch_count, history.history['loss'], 'r--')
    plt.plot(epoch_count, history.history['val_loss'], 'b-')

    plt.plot(epoch_count, history.history['acc'], 'gray')
    plt.plot(epoch_count, history.history['val_acc'], 'green')

    plt.legend(['Training Loss', 'Validation Loss','Training Acc', 'Validation Acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def Naive_Bayes(train_path,test_path):
    clf = MultinomialNB()

    X_train, Y_train = load_TrainData(train_path)
    X_test,X_test_PhraseID = load_TestData(test_path)
    tv = TfidfVectorizer(min_df=5,
                              max_df=0.5,
                              analyzer='word',
                              strip_accents='unicode',
                              ngram_range=(1,3),
                              sublinear_tf=True,
                              smooth_idf=True
                              )
    X_train_Vec = tv.fit_transform(X_train)
    X_test_Vec = tv.transform(X_test)

    clf = MultinomialNB()
    # train model
    clf.fit(X_train_Vec,Y_train)

    # save model
    outfile = open('classifier.pickle','wb') 
    pickle.dump(clf,outfile)

    # predict
    y_predict = clf.predict(X_test_Vec)

    #save result
    csvFile = open("Submission_result_Naive Bayes.csv", "w", newline = "")
    writer = csv.writer(csvFile)
    data = ["PhraseID", "Sentiment"]
    writer.writerow(data)
    for i in range(len(y_predict)):
        data = [X_test_PhraseID[i], y_predict[i]]
        writer.writerow(data)
    csvFile.close()

def Logistic_Regression(train_path,test_path):
    X_train, Y_train = load_TrainData(train_path)
    X_test,X_test_PhraseID = load_TestData(test_path)

    tv = TfidfVectorizer(min_df=5,
                              max_df=0.5,
                              analyzer='word',
                              strip_accents='unicode',
                              ngram_range=(1,3),
                              sublinear_tf=True,
                              smooth_idf=True
                              )
    X_train_Vec = tv.fit_transform(X_train)
    X_test_Vec = tv.transform(X_test)

    clf = LogisticRegression(solver='liblinear', multi_class='ovr')
    # train model
    clf.fit(X_train_Vec,Y_train)

    # save model
    outfile = open('classifier.pickle','wb') 
    pickle.dump(clf,outfile)

    # predict
    y_predict = clf.predict(X_test_Vec)

    #save result
    csvFile = open("Submission_result_Logictic Regression.csv", "w", newline = "")
    writer = csv.writer(csvFile)
    data = ["PhraseID", "Sentiment"]
    writer.writerow(data)
    for i in range(len(y_predict)):
        data = [X_test_PhraseID[i], y_predict[i]]
        writer.writerow(data)
    csvFile.close()

def Support_Vector_Classifier(train_path,test_path):
    X_train, Y_train = load_TrainData(train_path)
    X_test,X_test_PhraseID = load_TestData(test_path)

    tv = TfidfVectorizer(min_df=5,
                              max_df=0.5,
                              analyzer='word',
                              strip_accents='unicode',
                              ngram_range=(1,3),
                              sublinear_tf=True,
                              smooth_idf=True
                              )
    X_train_Vec = tv.fit_transform(X_train)
    X_test_Vec = tv.transform(X_test)

    clf = SVC(kernel='rbf')
    # train model
    clf.fit(X_train_Vec,Y_train)

    # save model
    outfile = open('classifier.pickle','wb') 
    pickle.dump(clf,outfile)

    # predict
    y_predict = clf.predict(X_test_Vec)

    #save result
    csvFile = open("Submission_result_Support Vector Classifier.csv", "w", newline = "")
    writer = csv.writer(csvFile)
    data = ["PhraseID", "Sentiment"]
    writer.writerow(data)
    for i in range(len(y_predict)):
        data = [X_test_PhraseID[i], y_predict[i]]
        writer.writerow(data)
    csvFile.close()

def K_Neighbor(train_path,test_path):
    X_train, Y_train = load_TrainData(train_path)
    X_test,X_test_PhraseID = load_TestData(test_path)

    tv = TfidfVectorizer(min_df=5,
                              max_df=0.5,
                              analyzer='word',
                              strip_accents='unicode',
                              ngram_range=(1,3),
                              sublinear_tf=True,
                              smooth_idf=True
                              )
    X_train_Vec = tv.fit_transform(X_train)
    X_test_Vec = tv.transform(X_test)

    clf = KNeighborsClassifier() 
    # train model
    clf.fit(X_train_Vec,Y_train)

    # save model
    outfile = open('classifier.pickle','wb') 
    pickle.dump(clf,outfile)

    # predict
    y_predict = clf.predict(X_test_Vec)

    #save result
    csvFile = open("Submission_result_KNeighbors Classifier.csv", "w", newline = "")
    writer = csv.writer(csvFile)
    data = ["PhraseID", "Sentiment"]
    writer.writerow(data)
    for i in range(len(y_predict)):
        data = [X_test_PhraseID[i], y_predict[i]]
        writer.writerow(data)
    csvFile.close()

if __name__ == "__main__":
    load_TrainData('train.tsv')
    # LSTM_network('train.tsv','test.tsv')
    # Naive_Bayes('train.tsv','test.tsv')
    # Logistic_Regression('train.tsv','test.tsv')
    # Support_Vector_Classifier('train.tsv','test.tsv')
    # K_Neighbor('train.tsv','test.tsv')
