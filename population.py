from genome import Genome
from network import Network
from config import Config
from colorama import *
import copy
import random
import math


class Population(object):



    def __init__(self):
        #Genomes is an array with every individuals in this particular generation
        self.genomes = []

    def findFittest(self,number=1):
        fittest = sorted(self.genomes, key=lambda x: x.fitness, reverse=True)
        while len(fittest) > number:
            fittest.pop()
        if number==1:
            #No array in this case
            fittest=fittest[0]
        return fittest

    def generateRandomPopulation(self):
        for i in range(Config.numberOfIndividuals):
            genome = generateRandomGenome()
            self.genomes.append(genome)



    def evolvePopulation(self):
        genomes = []
        elitismoffset = 0

        if Config.elitism:
            elite = self.findFittest(Config.numberofEliteWeKeep)
            elitismoffset = Config.numberofEliteWeKeep

        if Config.mutateElite:
            for genome in elite:
                genome.mutate(Config.mutationRate)

        for x in range (self.size() - elitismoffset - Config.numberofNewRandomGenomePerGeneration):
            if Config.selectionMethod=="tournamentSelection":
                if Config.FatherAlwaysElite:
                    if Config.FatherRandomElite:
                        if Config.numberofEliteWeKeep !=1:
                            father = randomGenome(elite)
                        else:
                            #There is only one elite so we don't bother taking a random one in a set of one...
                            father.self.findFittest()
                    else:
                        father = self.findFittest()
                else:
                    father = tournamentSelection(self)
                mother = tournamentSelection(self)

            if Config.selectionMethod=="rouletteWheelSelection":
                if Config.FatherAlwaysElite:
                    if Config.FatherRandomElite:
                        if Config.numberofEliteWeKeep !=1:
                            father = randomGenome(elite)
                        else:
                            #There is only one elite so we don't bother taking a random one in a set of one...
                            father = self.findFittest()
                else:
                    father = rouletteWheelSelection(self)
                mother = rouletteWheelSelection(self)


            newIndiv = self.crossover(father, mother)
            if Config.mutateNewBorn:
                newIndiv.mutate(Config.mutationRate)

            genomes.append(newIndiv)

        #Random genome
        for x in range (Config.numberofNewRandomGenomePerGeneration):
            newIndiv = generateRandomGenome()

            if Config.mutateNewBorn:
                newIndiv.mutate(Config.mutationRate)
            genomes.append(newIndiv)


        #Finally we add all elite at the beginning of the pool
        if Config.numberofEliteWeKeep !=1:
            genomes = elite + genomes
        else:
            genomes = genomes.append(elite)

        self.genomes = genomes


    def oldEvolvePopulationv2(self):
        genomes = []
        elite = self.findFittest(Config.numberofEliteWeKeep)
        #printc("Fittest : %s" % self.findFittest().genes,"red")
        #We keep n elite and we mutate them
        if Config.mutateElite==True:
            for genome in elite:
                genome.mutate(Config.mutationRate)
                genome.mutate(Config.mutationRate)
        genomes.extend(elite)
        print('\n' * 5)
        #Fill the table with baby of elite
        while len(genomes) < (Config.numberOfIndividuals - Config.numberofNewRandomGenomePerGeneration):
            gen1 = randomGenome(elite)
            gen2 = randomGenome(elite)
            newIndiv = self.crossover(gen1,gen2)
            newIndiv.mutate(Config.mutationRate)
            newIndiv.mutate(Config.mutationRate)
            genomes.append(newIndiv)

        #Random new genome
        while len(genomes) < Config.numberOfIndividuals:
            newIndiv = generateRandomGenome()
            newIndiv.mutate(Config.mutationRate)
            newIndiv.mutate(Config.mutationRate)
            genomes.append(newIndiv)

        self.genomes = genomes

    def oldEvolvePopulationv3(self):
        genomes = []
        elite = self.findFittest(self.numberofEliteWeKeep)
        #printc("Fittest : %s" % self.findFittest().genes,"red")
        #We keep n elite and we mutate them but slightlier than the other
        for genome in elite:
            genome.mutatev2("weights",self.mutationRate / 2)
            genome.mutatev2("biases",self.mutationRate / 2)
        genomes.extend(elite)
        print('\n' * 5)
        #Fill the table with baby of the elite
        while len(genomes) < (self.numberOfIndividuals - self.numberofNewRandomGenomePerGeneration):
            if self.selectionMethod=="tournamentSelection":
                if self.FatherAlwaysElite==True:
                    father = randomGenome(elite)
                else:
                    father = tournamentSelection(self)
                mother = tournamentSelection(self)

            if self.selectionMethod=="rouletteWheelSelection":
                if self.FatherAlwaysElite==True:
                    father = randomGenome(elite)
                else:
                    father = rouletteWheelSelection(self)
                mother = rouletteWheelSelection(self)

            newIndiv = self.crossoverv2(father, mother,"biases")
            newIndiv.mutatev2("weights",self.mutationRate)
            newIndiv.mutatev2("biases",self.mutationRate)
            #print "Genome added"
            genomes.append(newIndiv)
        #Random new genome
        while len(genomes) < self.numberOfIndividuals:
            newIndiv = generateRandomGenome()
            newIndiv.mutatev2("weights",self.mutationRate)
            newIndiv.mutatev2("biases",self.mutationRate)
            genomes.append(newIndiv)

        self.genomes = genomes

    def crossover(self,genome1, genome2):
        #Once in a while genome2 is not set, pull request if you found why
        if type(genome2)==type(None):
            genome2=randomGenome(self.genomes)

        if Config.crossoverVersion==1:
            genome = Genome()
            for i in range(genome1.size()):
                # Crossover
                if random.random() <= Config.uniformRate:
                    genome.setGene(i, genome1.getGene(i))
                else:
                    genome.setGene(i, genome2.getGene(i))
            return genome

        #I know it's cryptic...
        if Config.crossoverVersion==2:
            genome = Genome()
            genes1 = genome1.genes
            genes2 = genome2.genes

            if random.random() < 0.5:
                tmp = genes1
                genes1 = genes2[:]
                genes2 = tmp[:]

            finalweights = genes1[:-genome1.network.numberOfBiases]

            biases1 = genes1[genome1.network.numberOfWeights:]
            biases2 = genes2[genome2.network.numberOfWeights:]
            cutLocation = int(math.floor(len(biases1) * random.random()))
            part1 = biases1[:cutLocation]
            part2 = biases2[cutLocation:]
            #Form the final genes
            part1.extend(part2)
            finalbiases = part1

            finalweights.extend(finalbiases)
            genome.genes = finalweights[:]
            return genome

    def addGenome(self, genome):
        self.genomes.append(genome)

    def setGenomeFitness(self,i,fitness):
        self.genomes[i].fitness = fitness

    def getGenome(self,i):
        return self.genomes[i]

    def averageFitness(self):
        fitness = []
        for genome in self.genomes:
            fitness.append(genome.fitness)
        return sum(fitness) / float(len(fitness))

    def sumFitness(self):
        fitness = []
        for genome in self.genomes:
            fitness.append(genome.fitness)
        return sum(fitness)

    def size(self):
        return len(self.genomes)




def randomGenome(genomes):
    i = random.randint(0,len(genomes) - 1)
    return genomes[i]

def generateRandomGenome():
    net = Network(Config.networkArchitecture)
    genome = Genome(net)
    return genome

def printc(text,color):
    if color == "red":
         print(Fore.RED)
    if color == "blue":
         print(Fore.BLUE)
    if color == "green":
         print(Fore.GREEN)
    print(text)
    print(Style.RESET_ALL)

def rouletteWheelSelection(population):
    max     = sum([g.fitness for g in population.genomes])
    pick    = random.uniform(0, max)
    current = 0
    for genome in population.genomes:
        current += genome.fitness
        if current > pick:
            return genome

def tournamentSelection(population):
    #print "A new tournament has begun !!!"
    # Create a tournament population
    tournament = Population()
    # For each place in the tournament get a random individual
    for i in range(Config.tournamentSize):
        randomInt = random.randint(0,Config.numberOfIndividuals - 1)
        #print "individual added to the tournament, number %s" % randomInt
        #print('\n' * 3)
        tournament.addGenome(population.genomes[randomInt])

    # Get the fittest
    fittest = tournament.findFittest()
    #"Winner : %s" % fittest.genes
    return fittest
