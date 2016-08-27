import random
from config import Config
from network import Network

class Genome(object):
    networkArchitecture = [2,3,1]
    fitness = 0
    mutationMinimizer = 1


    def __init__(self,network=0):
        if network==0:
            self.network = Network(self.networkArchitecture)
        else:
            self.network = network

        self.genes = self.network.togenes()

    def clone(self):
        genome = Genome(self.network)
        return genome

    def getNetwork(self):
        return self.network
    def getGene(self,i):
        return self.genes[i]
    def setGene(self,i,value):
        self.genes[i]=value
    def size(self):
        return len(self.genes)

    def mutate(self, mutationRate):
        newgenes = []

        if Config.mutateVersion==1:
            for gene in self.genes:
                if random.random() <= mutationRate:
                    gene = random.random() / self.mutationMinimizer
                newgenes.append(gene)
            self.genes = newgenes
            return

        if Config.mutateVersion==2:
            weights = self.genes[:-self.network.numberOfBiases]
            biases = self.genes[self.network.numberOfWeights:]

            for weight in weights:
                if random.random() <= mutationRate:
                    weight += weight * (random.random() - 0.5) * 3 + (random.random() - 0.5)

            for bias in biases:
                if random.random() <= mutationRate:
                    bias += bias * (random.random() - 0.5) * 3 + (random.random() - 0.5)

            newgenes.extend(weights)
            newgenes.extend(biases)
            self.genes = newgenes
            return
