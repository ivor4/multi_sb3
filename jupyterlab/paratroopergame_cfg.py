import pygame
import math
import numpy as np

#Game Modes
GAME_MODE_NORMAL = 0
GAME_MODE_EXT_ACTION = 1

#Action values
#ACTION_NONE = 0
#ACTION_SHOOT = 1
#ACTION_ROTATE_RIGHT = 2
#ACTION_ROTATE_LEFT = 3


POOL_SIZE = 16

# Just their array index in _POSSIBLE_KEYS list
KEY_INDEX_JUMP = 0
KEY_INDEX_RIGHT = 1
KEY_INDEX_LEFT = 2



ACTION_LIST_INFO = [(KEY_INDEX_JUMP, 'Shoot'), (KEY_INDEX_RIGHT, 'Rotate Right'), (KEY_INDEX_LEFT, 'Rotate Left')]

#Values associated for keys are designed to be bit fields, so every value is power of 2 and can be mannipulated with binary operators easily
#Tuple is pygame keycode and associated power2 value for bitfield
POSSIBLE_KEYS = []
POSSIBLE_KEYS.append((pygame.K_SPACE, 0x1))
POSSIBLE_KEYS.append((pygame.K_RIGHT, 0x2))
POSSIBLE_KEYS.append((pygame.K_LEFT, 0x4))

ACTION_NONE = [False,False,False]

#Different object types
OBJ_TYPE_CANNON = 0
OBJ_TYPE_BULLET = 1
OBJ_TYPE_EXPLOSION = 2
OBJ_TYPE_AIRCRAFT = 3
OBJ_TYPE_PARATROOPER = 4
OBJ_TYPE_WALL = 5

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
CANNON_SIZE = [16,48]
CANNON_SEGMENTS = 3
CANNON_SEGMENT_SIZE = [CANNON_SIZE[0], CANNON_SIZE[1]//CANNON_SEGMENTS]
CANNON_SEGMENT_RADIUS = CANNON_SEGMENT_SIZE[1]
CANNON_SEGMENT_RADIUS_HALF = CANNON_SEGMENT_SIZE[1]/2.0

CANNON_MIN_ANGLE = 0.0
CANNON_MAX_ANGLE = math.pi
CANNON_ANGLE_AMPLITUDE = CANNON_MAX_ANGLE - CANNON_MIN_ANGLE

#180deg in 1 second
CANNON_ANGLE_ADVANCE = CANNON_ANGLE_AMPLITUDE/(1*EXPECTED_FPS)


#Game output
OUTPUT_SIZE_FACTOR = 16

#Xmin,XMax - X axis Region of interest of screen in pixels. Depending on this, observation will be bigger or smaller
REGION_OF_INTEREST = [0,1280]

#Output in X will be reduced according to factor
OUTPUT_NP_X_LENGTH = (REGION_OF_INTEREST[1] - REGION_OF_INTEREST[0]) // OUTPUT_SIZE_FACTOR

#Mlp will be used and not an image, Y axis is = number of aircraft lanes
OUTPUT_NP_Y_LENGTH = SCREEN_SIZE[1] // OUTPUT_SIZE_FACTOR

#Visual segments to help to know where cannon aims
LASER_SEGMENTS = 10

RADIAL_LASER_DISTANCE = ((SCREEN_SIZE[0]//2)//OUTPUT_SIZE_FACTOR)//LASER_SEGMENTS



#Game Difficulty

#Height at which Cannon is placed (the lower, the easier)
CANNON_HEIGHT = SCREEN_SIZE[1] * 5 // 6

BULLET_RELOAD_TIME_CYCLES = 10
BULLET_SPEED_ADVANCE_PER_CYCLE = 20

# Range between min and max cycles (will be used with random range)
AIRCRAFT_SPAWN_TIME = [60, 180]

# Aircraft speed range (random)
AIRCRAFT_SPEED_RANGE = [7, 14]

# Drop zones with min and max X each one
DROP_ZONES = [[100,SCREEN_SIZE[0]//2 - 100], [SCREEN_SIZE[0]//2 + 100, SCREEN_SIZE[0]-100]]

# Falling speed of paratrooper
PARATROOPER_FALL_SPEED = 3

# Max number of permitted paratroopers reach bottom part
MAX_PARATROOPERS_REACH_BOTTOM = 5


