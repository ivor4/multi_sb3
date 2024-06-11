import pygame
import math
import numpy as np

#Game Modes
GAME_MODE_NORMAL = 0
GAME_MODE_EXT_ACTION = 1



# Just their array index in _POSSIBLE_KEYS list
KEY_INDEX_UP = 0
KEY_INDEX_DOWN = 1
KEY_INDEX_RIGHT = 2
KEY_INDEX_LEFT = 3



ACTION_LIST_INFO = [(KEY_INDEX_UP, 'Up'), (KEY_INDEX_DOWN, 'Down'), (KEY_INDEX_RIGHT, 'Right'), (KEY_INDEX_LEFT, 'Left')]

#Values associated for keys are designed to be bit fields, so every value is power of 2 and can be mannipulated with binary operators easily
#Tuple is pygame keycode and associated power2 value for bitfield
POSSIBLE_KEYS = []
POSSIBLE_KEYS.append((pygame.K_UP, 0x1))
POSSIBLE_KEYS.append((pygame.K_DOWN, 0x2))
POSSIBLE_KEYS.append((pygame.K_RIGHT, 0x4))
POSSIBLE_KEYS.append((pygame.K_LEFT, 0x8))

ACTION_NONE = [False,False,False,False]

#Different object types
OBJ_TYPE_CROSSHAIR = 0
OBJ_TYPE_BULLET = 1


#Different ways to destroy an object
KILLED_NOT = 0
KILLED_WASTED = 1
KILLED_BINGO = 2
KILLED_NEUTRAL = 3

#Different types of shapes
SHAPE_RECTANGLE = 0
SHAPE_CIRCLE = 1

#Screen size
SCREEN_SIZE = [1280, 720]

SCREEN_RECT = pygame.Rect(0,0,SCREEN_SIZE[0],SCREEN_SIZE[1])


#Game time
ROUND_TIME_S = 120
EXPECTED_FPS = 60
TIME_PER_CYCLE = 1/EXPECTED_FPS
ROUND_CYCLES = ROUND_TIME_S * EXPECTED_FPS

#Cannon size
CROSSHAIR_SIZE = [64,64]


#Game output
OUTPUT_SIZE_FACTOR = 16

#Xmin,XMax - X axis Region of interest of screen in pixels. Depending on this, observation will be bigger or smaller
REGION_OF_INTEREST = [0,1280]

#Output in X will be reduced according to factor
OUTPUT_NP_X_LENGTH = (REGION_OF_INTEREST[1] - REGION_OF_INTEREST[0]) // OUTPUT_SIZE_FACTOR

#Mlp will be used and not an image, Y axis is = number of aircraft lanes
OUTPUT_NP_Y_LENGTH = SCREEN_SIZE[1] // OUTPUT_SIZE_FACTOR


#Game Difficulty

# Crosshair move speed
CROSSHAIR_SPEED = 4

# Bullet speed
BULLET_SPEED = 3.9

# Max time outside crosshair
MAX_TIME_OUTSIDE = 5.0




