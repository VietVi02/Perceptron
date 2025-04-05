#-------------------------------------------------------------------------
# AUTHOR: Viet Tran
# FILENAME: perceptron.py
# SPECIFICATION: Train a Single Layer Perceptron and Multi-Layer Perceptron to classify optically recognized handwritten digits.
# FOR: CS 4210- Assignment #3
# TIME SPENT: ~4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

# Initialize variables to track the best accuracies
best_perceptron_acc = 0
best_mlp_acc = 0
best_perceptron_params = None
best_mlp_params = None

for learning_rate in n: #iterates over n

    for shuffle in r: #iterates over r

        #iterates over both algorithms
        clf_perceptron = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
        clf_perceptron.fit(X_training, y_training)

        for algorithm in ['Perceptron', 'MLP']: #iterates over the algorithms

            #Create a Neural Network classifier
            if algorithm == 'Perceptron':
                 # Create a Perceptron classifier
                clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
            elif algorithm == 'MLP':
                # Create a MLP classifier
                clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, 
                                    hidden_layer_sizes=(25,), shuffle=shuffle, max_iter=1000)
            
            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            # Make predictions and calculate accuracy
            correct_predictions = 0
            for x_test_sample, y_test_sample in zip(X_test, y_test):
                prediction = clf.predict([x_test_sample])
                if prediction == y_test_sample:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(y_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            if algorithm == 'Perceptron' and accuracy > best_perceptron_acc:
                best_perceptron_acc = accuracy
                best_perceptron_params = {'learning_rate': learning_rate, 'shuffle': shuffle}
                print(f"Highest Perceptron accuracy so far: {best_perceptron_acc}, Parameters: learning rate={learning_rate}, shuffle={shuffle}")
            
            elif algorithm == 'MLP' and accuracy > best_mlp_acc:
                best_mlp_acc = accuracy
                best_mlp_params = {'learning_rate': learning_rate, 'shuffle': shuffle}
                print(f"Highest MLP accuracy so far: {best_mlp_acc}, Parameters: learning rate={learning_rate}, shuffle={shuffle}")

# Final output after completing the iterations
print(f"Final Highest Perceptron accuracy: {best_perceptron_acc}, Parameters: learning rate={best_perceptron_params['learning_rate']}, shuffle={best_perceptron_params['shuffle']}")
print(f"Final Highest MLP accuracy: {best_mlp_acc}, Parameters: learning rate={best_mlp_params['learning_rate']}, shuffle={best_mlp_params['shuffle']}")











