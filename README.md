# FlapAI: Genetic algorithm learning to play Flappy Bird (Python)
### Introduction
---
**FlapAI** is a genetic algorithm coded in Python teaching itself  how to play Flappy Bird. It can reach score of 1000+ pipes in less than 100 generations. It is based on [FlapPyBird](https://github.com/sourabhv/FlapPyBird). All settings can be found in **config.py**. Neural networks are saved in a json file for further evaluation. Each time you run FlapAI, statistics and the neural network of the best performing bird are saved in the directory **save/**
#### Features
---
- Customizable genetic algorithm
- Speed up *(FPS can be set)*
- Statistics
- Running a single Neural Network from a json file multiple time
- Cool ASCII art
### Requirements
---
- pygame
- numpy
- matplotlib
- colorama *(cool colored **print**)*


### Usage
---
If you want to train birds with the parameters in config.py just run FlapAI.
```sh
python flapai.py
```
To evaluate a single Neural Network use **-evaluate** (some can be found in **/interestingannsave**)
```sh
python flapai.py -evaluate neuralnetwork.json
```
If you want to see the statistics of a past experiment use **-stats**
```sh
python flapai.py -stats save/2016-08-27_17:16:04
```
### Keyboard commands
---
Press **UP ARROW** while focusing the game window to speed up the game **(if the screen is frozen it means that pygame don't redraw the window anymore which speed the algorithm a lot)**
#
Press **DOWN ARROW** while focusing the game window to slow down the game
#
Press **ESC** while focusing the game window to stop the algorithm and show the statistics **(don't use CTRL+C in the terminal)**

### Screenshots
---
![FlapAI](screenshots/1.png)
![Ninja!](screenshots/output.gif)
