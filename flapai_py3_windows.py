#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
FlapAI: Genetic algorithm learning to play Flappy Bird (Python)
Original version: https://github.com/justinglibert/flapai/
The original version is running under python2 and Linux.

This new version is running under python3 and Windows.
Author: rex_huang61
E-mail: 442193160@qq.com
Github: https://github.com/zhiruihuang/flapai/
Date: 2017/11/30
"""
print(__doc__)

from itertools import cycle
import random
import sys
import time
import datetime
import logging
import os

import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import json

from genome import Genome
from network import Network
from population import Population
from config import Config

import pickle
import numpy as np
import subprocess

from colorama import *



# today = "save/" + str(datetime.date.today()) + "_" + time.strftime("%X") # Linux
today = "/save/" + str(datetime.date.today()) + "_" + time.strftime("%H%M%S") # Windows

if not os.path.exists(today):
    os.makedirs(today)

savestat = True
fpsspeed=3
FPS = 4000

bestFitness = 0
fitnessovergeneration = []
fittestovergeneration = []

#detectionOffset = 36
detectionOffset = 40

SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image and hitmask  dicts
IMAGES, HITMASKS = {}, {}
DIEIFTOUCHTOP = True
# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

asciiart="""
______ _              ___  _____
|  ___| |            / _ \|_   _|
| |_  | | __ _ _ __ / /_\ \ | |
|  _| | |/ _` | '_ \|  _  | | |
| |   | | (_| | |_) | | | |_| |_
\_|   |_|\__,_| .__/\_| |_/\___/
              | |
              |_|    """

def printc(text,color):
    if color == "red":
         print(Fore.RED)
    if color == "blue":
         print(Fore.BLUE)
    if color == "green":
         print(Fore.GREEN)
    print(text)
    print(Style.RESET_ALL)

def main():





    """Parse option"""
    if len(sys.argv) != 1:
        #Evaluate a single genome
        if str(sys.argv[1])=="-evaluate":
            initPygame("FlapAI: Evaluating")
            print(str(sys.argv[2]))
            net = load(str(sys.argv[2]))
            genome = Genome(net)
            global savestat
            savestat = False
            fitness = []
            for i in range(100):
                fit = playGame(genome)
                fitness.append(fit.fitness)
                print("fitness : %s " % fit.fitness)

            average = sum(fitness) / float(len(fitness))
            printc("Average fitness : %s" % average,"red")
            pygame.quit()
            sys.exit()
        #Show the stat of an experiment
        if str(sys.argv[1])=="-stats":
            showStat(str(sys.argv[2]))



    #No argument, starting FlapAI
    initPygame("FlapAI: Learning")

    """Init the population"""
    bestFitness = 0
    population = Population()
    population.generateRandomPopulation()
    generation = 1
    maxgeneration = Config.maxGeneration

    lastgenerationaveragefitness = 0
    #Main Loop
    while generation <= maxgeneration :
        birdnmbr = 1

        for i in range(population.size()):

            genome = playGame(population.getGenome(i))
            population.setGenomeFitness(i,genome.fitness)
            informationforscreen = {
            'generation' : generation,
            'birdnumber' : birdnmbr,
            'lastfitness' : genome.fitness,
            'lastgenerationaveragefitness' : lastgenerationaveragefitness,
            'bestfitness' : bestFitness
            }
            updateScreen(informationforscreen)
            if genome.fitness > bestFitness:
                global bestFitness
                bestFitness = genome.fitness
                genome.network.save(today + "/bestfitness.json")
            birdnmbr += 1


        global fitnessovergeneration
        fitnessovergeneration.append(population.averageFitness())

        lastgenerationaveragefitness = population.averageFitness()

        global fittestovergeneration
        fittestovergeneration.append(population.findFittest().fitness)
        #Evolve the population
        population.evolvePopulation()
        generation += 1


def initPygame(screencaption):
        global SCREEN, FPSCLOCK
        pygame.init()
        init()
        FPSCLOCK = pygame.time.Clock()
        SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        pygame.display.set_caption(screencaption)

        # numbers sprites for score display
        IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # base (ground) sprite
        IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

def playGame(genome):

    """Info pour le jeux"""
    # select random background sprites
    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )
    """Info pour lancer le jeux sans le message au depart"""

    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)
    playerShmVals = {'val': 0, 'dir': 1}
    basex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    movementInfo = {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }
    #Update the network with current genes
    genome.network.fromgenes(genome.genes)
    crashInfo = mainGame(movementInfo,genome)
    #fitness = showGameOverScreen(crashInfo)
    genome = crashInfo['genome']

    if Config.fitnessIsScore:
        genome.fitness = crashInfo['score']

    if genome.fitness < 0:
        genome.fitness = 0
    return genome

def mainGame(movementInfo,genome):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    framesurvived = 0

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                #Store Stat
                if savestat==True:
                    reportStat()

                pygame.quit()
                if savestat==True:
                    showStat(today)
                else:
                    sys.exit()
            if event.type == KEYDOWN and event.key == K_UP:
                if fpsspeed < 4:
                    global fpsspeed
                    fpsspeed += 1
            if event.type == KEYDOWN and event.key == K_DOWN:
                if fpsspeed != -2:
                    global fpsspeed
                    fpsspeed -= 1
        #Evaluate the NN
        if playerx < lowerPipes[0]['x'] + detectionOffset:
            nextPipe = lowerPipes[0]
        else:
            nextPipe = lowerPipes[1]

        nextPipeY = float(SCREENHEIGHT - nextPipe['y']) / SCREENHEIGHT

        playerYcorrectAxis = float(SCREENHEIGHT - playery) / SCREENHEIGHT
        distanceBetweenPlayerAndNextPipe = float(nextPipe['x'] - playerx)/ SCREENWIDTH

        NNinput = np.array([[playerYcorrectAxis],[nextPipeY]])

        NNoutput = genome.network.feedforward(NNinput)

        if NNoutput > 0.5:
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True

        info = {'playery': playerYcorrectAxis, 'pipey': nextPipeY, 'distance': distanceBetweenPlayerAndNextPipe}


        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        if crashTest[0] or playery < 5:
            genome.fitness = framesurvived
            return {
                'score': score,
                'genome': genome
            }

        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        playerHeight = IMAGES['player'][playerIndex].get_height()
        if playery > 5:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        if Config.debug:
            displayInfo(info)

        framesurvived += 1
        SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))


        global FPS
        if fpsspeed==4:
            #No FPS clock ticking, may be instable
            continue
        if fpsspeed==3:
            FPS=4000
        if fpsspeed==2:
            FPS=400
        if fpsspeed==1:
            FPS=40
        if fpsspeed==0:
            FPS=30
        if fpsspeed==-1:
            FPS=15
        if fpsspeed==-2:
            FPS=3

        pygame.display.update()
        FPSCLOCK.tick(FPS)

def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1

def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]

def displayInfo(info):
    ###Display useful info : the input for the ANN
    myfont = pygame.font.Font(None, 30)
    # render text
    playery = str(info['playery'])
    tubey = str(info['pipey'])
    distance = str(info['distance'])

    labelplayery = myfont.render(playery,1,(255,255,0))
    labeltubey = myfont.render(tubey,1,(0,255,255))
    labeldistance = myfont.render(distance,1,(255,255,255))

    SCREEN.blit(labelplayery, (SCREENWIDTH / 2 - 100, SCREENHEIGHT * 0.7))
    SCREEN.blit(labeltubey, (SCREENWIDTH / 2  - 100, SCREENHEIGHT * 0.8))
    SCREEN.blit(labeldistance, (SCREENWIDTH / 2 - 100, SCREENHEIGHT * 0.9))

def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

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

def reportStat():
    with open(today + '/fitnessovergeneration.dat', 'wb') as handle:
        pickle.dump(fitnessovergeneration, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(today + '/fittestovergeneration.dat', 'wb') as handle:
        pickle.dump(fittestovergeneration, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(today + '/bestfitness.dat', 'wb') as handle:
        pickle.dump(bestFitness, handle, protocol=pickle.HIGHEST_PROTOCOL)

def updateScreen(info):
    #Clear the screen
    # subprocess.call(["printf", "\033c"]) # Linux
    subprocess.call(["cls"], shell=True) # Windows
    #Print asciiart
    printc(asciiart,"green")

    if info["generation"] > 1:
        print("----Last generation----")
        printc("Average fitness: %s" % str(info["lastgenerationaveragefitness"]), "blue")
        print("-----------------------")
    if info["birdnumber"] > 1:
        printc("Last Fitness: %s" % str(info["lastfitness"]), "green")

    printc("Best Fitness: %s" % str(info["bestfitness"]),"red")
    print("----Status----")
    printc("Generation number : %s/%s" % (str(info["generation"]),str(Config.maxGeneration)),"blue")
    printc("Bird number: %s/%s" % (str(info["birdnumber"]), str(Config.numberOfIndividuals)),"blue")

def showStat(folder):

    fitnessovergeneration = pickle.load(open(folder + '/fitnessovergeneration.dat', 'rb'))
    fittestovergeneration = pickle.load(open(folder + '/fittestovergeneration.dat', 'rb'))
    bestfitness = pickle.load(open(folder + '/bestfitness.dat', 'rb'))

    #Clear the screen
    # subprocess.call(["printf", "\033c"]) # Linux
    subprocess.call(["cls"], shell=True) # Windows
    printc("Statistics of %s" % folder,"blue")
    printc("-" * 20,"green")
    print("Number of generation: %s" % len(fittestovergeneration))
    printc("Best Fitness: %s" % bestFitness,"red")

    plt.figure(1)
    plt.plot(fittestovergeneration)
    plt.plot(fitnessovergeneration)
    plt.show()
    sys.exit()




if __name__ == '__main__':
    main()
