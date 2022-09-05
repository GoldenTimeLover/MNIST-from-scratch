#ID: 318423415
import pandas as pd
import numpy as np
import mnist_loader as mnl
import matplotlib.pyplot as plt

"""
The sigmoid function.
"""
def sigmoid(z):

    return 1/(1+np.exp(-z))



"""
class for a neural network designed to predict and categorize the
MNIST dataset.
"""
class NeuralNetwork:


    """
    constructor for NeuralNetwork if learning rate and epochs not provided
    has default values
    """
    def __init__(self,learningRate=0.1,epochs=10) -> None:

        #initializing variables
        self.learningRate = learningRate
        self.epochs = epochs

        #setting up our random weights
        self.weights_input_to_hidden_layer = np.random.uniform(-1, 1, (40, 784))
        self.weights_hidden_layer_to_output = np.random.uniform(-1, 1, (10, 40))

        #setting up our biases
        self.bias_input_to_hidden_layer = np.zeros((40, 1))
        self.bias_hidden_layer_to_output = np.zeros((10, 1))




        
    """
    fit(self,X,y) - Trains a neural network on X y.
    X - X is a matrix containing all or a part of the MNIST dataset.
    y - y is a vector with the corresponding labels.
    """
    def fit(self,X,y) -> None:
        for epoch in range(self.epochs):

            for img, label in zip(X, y):
                img.shape += (1,)

                #turn label int into label matrix\vector
                temp = np.zeros(10)
                temp[label] = 1
                temp.shape += (1,)
                label = temp


                ####
                # Forward Propagation
                ####

                hidden_layer = self.bias_input_to_hidden_layer + self.weights_input_to_hidden_layer @ img
                hidden_layer = sigmoid(hidden_layer)

                output_layer = self.bias_hidden_layer_to_output + self.weights_hidden_layer_to_output @ hidden_layer
                output_layer = sigmoid(output_layer)

                ####
                # Back Propagation
                ####

                errDer = output_layer - label
                self.weights_hidden_layer_to_output += -self.learningRate * errDer @ np.transpose(hidden_layer)
                self.bias_hidden_layer_to_output += -self.learningRate * errDer

                errHid = np.transpose(self.weights_hidden_layer_to_output) @ errDer * (hidden_layer * (1 - hidden_layer))
                self.weights_input_to_hidden_layer += -self.learningRate * errHid @ np.transpose(img)
                self.bias_input_to_hidden_layer += -self.learningRate * errHid



    """
    predict(self,X) - Computes the output of the trained network on the examples in X.
    returns the prediction of the network on the example X. an int 0-9
    """
    def predict(self,X) -> int:


        output = []
        for x in X:
            x.shape += (1,)

            # feed through the first layer
            hidden_layer = self.bias_input_to_hidden_layer + self.weights_input_to_hidden_layer @ x.reshape(784, 1)
            hidden_layer = sigmoid(hidden_layer)

            # feed from the hidden layer to the output layer
            output_layer = self.bias_hidden_layer_to_output + self.weights_hidden_layer_to_output @ hidden_layer
            output_layer = sigmoid(output_layer)


            #return max value neuron
            output.append(output_layer.argmax())
        
        return output
        

    """
    score(self,X,y) - Computes the average number of examples in X that the trained network classifees
    incorrectly.
    """
    def score(self,X,y) -> float:

        successCounter = 0
        errorCounter = 0
        
        results = self.predict(X)


        for res, label in zip(results, y):
            if(res == label):
                successCounter +=1
            else:
                errorCounter +=1
        

        successRate = successCounter / (successCounter+errorCounter)
        errorRate = errorCounter / (successCounter+errorCounter)

        print("the error rate is {}".format(errorRate))
        print("the success rate is {}".format(successRate))
        return errorRate






if __name__ == "__main__":

    
    print("loading data from file")

    #load mnist data from file 'mnist.pkl.gz'
    train , validate , test = mnl.load_data()


    #seperate the train data from the labels
    X = train[0]
    y = train[1]

    #seperate the test data from the labels
    testX = test[0]
    testy = test[1]

    #initializing the NN
    nn = NeuralNetwork()

    print("training neural network (may take a few moments...)")
    #training
    nn.fit(X,y)

    print("score:")
    #scoring
    nn.score(testX,testy)

