

#### Libraries
# Standard library
import random
import os

# Third-party libraries
import numpy as np
import json

class Network(object):

    def __init__(self, sizes, weights=0, biases=0):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if weights==0:
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = weights;

        if biases==0:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        else:
            self.biases = biases;
        """ number Of Biases and weights """
        self.numberOfBiases = sum(self.sizes[1:])
        self.numberOfWeights = 0
        for i in range (1, len(self.sizes)):
            self.numberOfWeights += self.sizes[i] * self.sizes [i-1]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input.
           Attention : a doit etre sous le format [[1],[2],...]
           donc de dimension (n,1) ou n est le nombre
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def getweights(self, layer, neuroninput, neuronoutput):
        return self.weights[layer][neuronoutput][neuroninput]

    def setweights(self, layer, neuroninput, neuronoutput, value):
        self.weights[layer][neuronoutput][neuroninput] = value

    def getbiases(self, layer, neuronoutput):
        return self.biases[layer][neuronoutput]

    def setbiases(self, layer, neuronoutput, value):
        self.biases[layer][neuronoutput] = value

    def togenes(self):
        genes = []
        for x in range(0, (self.num_layers - 1)):
            for y in range(0, (self.sizes[x + 1])):
                    genes.extend(self.weights[x][y])
        for x in range(len(self.biases)):
            for element in self.biases[x]:
                genes.extend(element)
        return genes

    def fromgenes(self,genes):
        weights1D = genes[:-self.numberOfBiases]
        biases1D = genes[self.numberOfWeights:]
        i = 0
        weights = []
        for x in range(0,(self.num_layers - 1)):
            layer = np.zeros((self.sizes[x + 1],self.sizes[x]),dtype=np.float64)
            for y in range(0, (self.sizes[x + 1] )):
                for z in range(0,self.sizes[x] ):
                    weight = weights1D[i]
                    layer[y,z] = weight
                    i += 1
            weights.append(layer)

        """ Now we parse the biases """
        biases = []
        i = 0
        for x in range(1,(self.num_layers)):
            biase = np.zeros((self.sizes[x],1),dtype=np.float64)
            for y in range(0, (self.sizes[x])):
                biase[y,0] = biases1D[i]
                i += 1
            biases.append(biase)

        self.weights = weights
        self.biases = biases

    def save(self, filename):
        """Save the neural network to the file filename """

        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        f = open(filename, "wb")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def sigmoid(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))
