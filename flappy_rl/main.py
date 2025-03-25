import pygame
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE

game = FlappyBird()
env = PLE(game, fps=30, display_screen=True)
env.init()

for _ in range(50):
    env.act(env.getActionSet()[0])  # Just do "nothing" for a few frames
