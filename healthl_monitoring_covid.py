# Packages importings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random 
import math
import mpmath as mp 
#we can import also mpmath for working with bigger values
# the data used in this program can be found at this link
#
class dataCleaningandSplitting:
    """_summary_
    in this module we clean the data convert from string to float and fron integers to float
    We also preform data screening, at the same time we split the data into train and test datasets
    Returns:
        array_list: training _set_ and testing sets are all matrices 
    """
    global minmax
    minmax= list()
    def __init__(self,data) -> None:
        """calling the data variable

        Args:
            data (panda.DataFrame): A csv file that is converted into dataframe using pandas
        """
        self.data=data
    def dataCleaning(self):
        data.info()
        data.replace({'Negative': 0,'Positive': 1},inplace=True)
        #data.dropna() # we do not need this as our data does not contain any kind null data
        data.drop('ID', inplace=True, axis=1)
    def plotting(self):
        x=[]
        y=[]
        x=list(data["X3"])
        y=list(data['Y1'])
        print(x)
        print(y)
        #data.plot(x,y)
        #plt.show()
    #max and min in the dataset
    def dataset_minmax(self):
        stats=[]
        for column in (data.columns):
            stats.append([min(data[column]),max(data[column])])
        return stats
    # Rescale dataset columns to the range 0-1
    def normalize_dataset(self):
        minmax=self.dataset_minmax()
        #convertinf all the data integers into floats
        for column in data.columns:
            data[column] = data[column].astype(float)
        """figure1=plt.figure()
        plt.plot(list(data["X5"]),list(data['Y2']))
        plt.xlabel("floor area")
        plt.ylabel=("heat load")
        plt.title("energy effciency")
        plt.grid()
        plt.show()"""
        #data.plot(kind='scatter', x=data["X3"],y=data['Y1'])
        for x in range(len(data)):
            for y in range(len(data.columns)):
                data.loc[x,data.columns[y]] = (data.loc[x,data.columns[y]] - minmax[y][0]) / (minmax[y][1] - minmax[y][0])
        figure2=plt.figure()
        """plt.plot(list(data["X5"]),list(data['Y2']))
        plt.xlabel("floor area")
        plt.ylabel=("heat load")
        plt.title("energy effciency")
        plt.grid()
        plt.show()"""
    def trainig_data(self):
        self.normalize_dataset()
        trainingSet=np.matrix(data.iloc[:5000,:])
        trainingSet=np.asfarray(trainingSet, int)
        return  trainingSet
    def testing_data(self):
        self.normalize_dataset()
        testingSet=np.matrix(data.iloc[5000:,:])
        testingSet=np.asfarray(testingSet, int)
        return testingSet
    
    def testing(self):
        self.normalize_dataset()
        testingSet=np.matrix(data)
        testingSet=np.asfarray(testingSet, int)
        return testingSet
class Backpropagation:
    def initialize(self, nInputs, nHidden, nOutputs):
        network = list()
        hiddenLayer = [{'weights': [16.50829339484901, -0.840038038553279, -7.0773890540620235, -4.039767392157919], 'output': 1.2154468557159535, 'delta': 0.12503613848642967}\
                       ,{'weights': [20.757486137333718, -14.965733193161565, -9.22012233125151, -2.9364187648009388], 'output': 0.0012808967163093683, 'delta': -9.491618062656173e-05}\
                       ,{'weights': [33.09105477949129, 13.930561795729258, -14.905220767374937, -18.711095382208963], 'output': 2.328441712701386, 'delta': -0.03018894746022163}]
        network.append(hiddenLayer)
        outputLayer = [{'weights': [-1.9161323893447022, 0.798838610624906, 0.3603547746501209, 0.5479292191264005], 'output': 0.3301265646923651, 'delta': -0.09282077873320603}]
        network.append(outputLayer)
        return network

    # Propagate forward
    def activate(self, inputs, weights):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self, activation):
        return np.log(1.0+np.exp(activation))#np.exp(activation) / np.sum(np.exp(activation))  #math.tanh(activation)#1.0 / (1.0 + math.exp(-activation))

    def forwardPropagate(self, network, row):
        inputs = row
        for layer in network:
            newInputs = []
            for neuron in layer:
                activation = self.activate(inputs, neuron['weights'])
                neuron['output'] = self.transfer(activation)
                newInputs.append(neuron['output'])
            inputs = newInputs
        return inputs

    # Propagate backwards
    def transferDerivative(self, output):
        #1-output**2#
        return (np.exp(output)-1)/(np.exp(output))#1/(1+mp.exp(-activation))#output * (1.0 - output)

    def backwardPropagateError(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if (i != len(network) - 1):
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transferDerivative(neuron['output'])

    # For train network
    def updateWeights(self, network, row, learningRate, nOutputs):
        nOutputs = nOutputs * -1
        for i in range(len(network)):
            inputs = row[:nOutputs]
            if (i != 0):
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(network[i])):
                    neuron['weights'][j] += learningRate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += learningRate * neuron['delta']

    def updateLearningRate(self, learningRate, decay, epoch):
        return learningRate * 1 / (1 + decay * epoch)

    def trainingNetwork(self, network, train, learningRate, nEpochs, nOutputs, expectedError):
        sumError = 10000.0
        for epoch in range(nEpochs):
            if (sumError <= expectedError):
                break
            if(epoch % 50 == 0):
                learningRate = self.updateLearningRate(learningRate, learningRate/nEpochs, float(epoch))

            sumError = 0
            for row in train:
                outputs = self.forwardPropagate(network, row)
                expected = self.getExpected(row, nOutputs)
                sumError += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backwardPropagateError(network, expected)
                self.updateWeights(network, row, learningRate, nOutputs)
            print('> epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learningRate, sumError))

    def getExpected(self, row, nOutputs):
        expected = []
        for i in range(nOutputs):
            temp = (nOutputs - i) * - 1
            expected.append(row[temp])
        return expected
    # For predict result
    def predict(self, network, row):
        outputs = self.forwardPropagate(network, row)
        return outputs
    def predictY1(self, network, row):
        outputs = self.forwardPropagate(network, row)
        return outputs
######################################################################################
data=pd.read_csv("Covid_data.csv")
load_data=dataCleaningandSplitting(data)
load_data.dataCleaning()
######################################################################################
#back propagation 
nOutputs =int(input('Insert the number Neurons into Output Layer: '))
nEpochs = int(input('Insert the number of Epochs: '))
nHiddenLayer =int(input('Insert the number Neurons into Hidden Layer: '))
learningRate = float(input('Insert Learning Rate: '))
expectedError = float(input('Insert Expected Error: '))
###
backpropagation = Backpropagation()
nInputs = len(dataCleaningandSplitting(data).trainig_data()[0]) - nOutputs
network = backpropagation.initialize(nInputs, nHiddenLayer, nOutputs)
#backpropagation.forwardPropagate(network,nInputs)
backpropagation.trainingNetwork(network, dataCleaningandSplitting(data).trainig_data(), learningRate, nEpochs, nOutputs, expectedError)
def writeWieghtsToFile(network):
    """_summary_

    Args:
        network (list_of_list): this contains the whole parameters of the network liek the \
            weights, the inputs, outputs and so on
    """
    my_df = pd.DataFrame(network)
    my_df.to_csv('my_array.csv',header = False, index= False)
writeWieghtsToFile(network)
input('\nPress enter to view Result...')
########################################################################################
Y1=list()
for row in dataCleaningandSplitting(data).testing():
    Y1.append(backpropagation.predictY1(network, row))
fig, ax = plt.subplots()
plt.title("Covid 19")
ax.plot(list(data['Result']), color = 'green', label = 'Expected Result')
ax.plot(Y1, color = 'red', label = 'Simulated Result')
ax.legend(loc = 'upper left')
plt.grid()
########################################################################################
error=0
accuracy=0
##testing the the network
correct=0
false_postive=0
false_negative=0
"""for row in dataCleaningandSplitting(data).testing_data():
    prediction = backpropagation.predict(network, row)
    expected=backpropagation.getExpected(row, nOutputs)
"""
print(Y1)
for i in range(len(Y1)):
  prediction=Y1[i]
  expected=data['Result'][i]
  if prediction[0]>0.15:
        prediction[0]=1
  else:
    prediction[0]=0
  if prediction==expected:
    correct+=1
  elif prediction[0]==0 and expected==1:
    false_negative+=1
  elif prediction[0]==1 and expected==0:
    false_postive+=1
accuracy=(correct)/(correct+false_negative+false_postive) *100
print("the total number of correct case are: {}".format(correct),"\n")
print("the number of false postive cases: {}, and the false negative cases: {},\n \
while the accuracy is {} %".format(false_postive, false_negative,accuracy))




"""for row in dataCleaningandSplitting(data).testing_data():
    prediction = backpropagation.predict(network, row)
    # print('Input =', (row), 'Expected = ', backpropagation.getExpected(row, nOutputs), 'Result =', (prediction))
    print('Expected = ', backpropagation.getExpected(row, nOutputs), 'Result =', (prediction))
    error=abs(backpropagation.getExpected(row, nOutputs)[0]-prediction)
    accuracy= (prediction/backpropagation.getExpected(row, nOutputs)[0])*100
print("error: {}, accuracy:{}".format(error,accuracy))"""