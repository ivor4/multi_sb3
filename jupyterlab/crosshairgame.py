import random
import numpy as np
from crosshairgame_objects import *
from crosshairgame_cfg import *
import pygame


class GameSystem:
    def __init__(self, caption, mode, render_mode):
        self.Caption = caption
        self.Mode = mode
        self.render_mode = render_mode

        if(pygame.get_init()):
            if(render_mode == 'human'):
                if(not pygame.display.get_init()):
                    pygame.display.init()
                    while(not pygame.display.get_init()):
                        pass
                pygame.display.set_caption(self.Caption)
                self.Screen = pygame.display.set_mode((SCREEN_SIZE[0], SCREEN_SIZE[1]))
                self.Clock = pygame.time.Clock()
                self.Font = pygame.font.SysFont(None, 24)
            else:
                self.Screen = None
                self.Clock = None
                self.Font = None
            
            self.OK = True
        else:
            self.OK = False
            raise Exception('Pygame not initialized')


    def reset(self, seed = 0):

        if(self.OK):

            # This list is the most important in game, as only instances listed here will be updated and/or rendered. An instance outside this list is unable to do anything
            self.GameObjects = []

            # This is a fast-access to bullets (only bullets), to make as fast as possible detection of collission iterating against this list
            self.ActiveBullets = []
            self.ActiveCrosshairs = []


            # Initialize action box to all 0
            self.InputActions = np.zeros(len(POSSIBLE_KEYS))

            # Own Random instance (for seeding purposes and in order to not to interact with global library)
            self.Random = random.Random(seed)

            # Pass GameObjects list to class static variable (this make possible automatic register/de-register when creating or destroying)
            GameObject.GameObjects = self.GameObjects
            GameObject.Random = self.Random

            # Pass Active Bullets list to class static variable (this make possible automatic register/de-register when creating or destroying)
            GameObject.ActiveBullets = self.ActiveBullets
            GameObject.ActiveCrosshairs = self.ActiveCrosshairs

            

            # Create Crosshair instance
            self.BulletInstance = Bullet()
            
            self.BulletInstance.ReCreate(pygame.Vector2(SCREEN_SIZE[0]//2, SCREEN_SIZE[1]//2), pygame.Vector2(0.0,0.0))
            
            #Randomize initial speed
            self.NextBulletChangeTimeout = 0.0
            self.DiscountBulletTime()
            
            self.CrosshairInstance = CrossHair()
            self.CrosshairInstance.ReCreate(pygame.Vector2(SCREEN_SIZE[0]//2, SCREEN_SIZE[1]//2))
            
            
                

            # Prepare elapsed time
            self.ElapsedTime = 0.0
            self.InsideTime = 0.0
            self.OutsideTime = 0.0

            #Create output observation array
            self.OutputObs = np.zeros((OUTPUT_NP_Y_LENGTH, OUTPUT_NP_X_LENGTH, 1))

            # Running game = True (by the moment)
            self.Running = True

            # Score initialization
            self.Score = 0


            # Neutral pressed keys in bit field is 0
            self.DownKeys = 0x0
            
            info = {}
            info['BulletPosition'] = [0, 0]
            info['CrosshairPosition'] = [0, 0]
            info['DeltaPosition'] = [0,0]
            info['ElapsedTime'] = 0
            info['InsideTime'] = 0
            info['OutsideTime'] = 0

            # Launch a first black screen to window
            if(self.Screen != None):
                self.Screen.fill('black')
                pygame.display.flip()

            return self.OutputObs, info
        else:
            raise Exception("Game cannot start")
        
    # Information
    def isRunning(self):
        return self.Running

    # Basic close operation (terminates game and also pygame system and window)
    def close(self):
        if(self.Running):
            self.Running = False
        if(self.render_mode == 'human'):
            pygame.display.quit()

    # Get pressed keys and determine an action
    def _KeyDetection(self):
        keys = pygame.key.get_pressed()

        # Neutral pressed keys in bit field is 0
        actualDownKeys = 0x0
        
        # Search along all possible declared keys, the ones which were pressed by doing OR operation, as they are power of 2, they will not overlap
        for possible_key in POSSIBLE_KEYS:
            if keys[possible_key[0]]:
                actualDownKeys |= possible_key[1]

        # XOR between actual used keys and last cycle will give the keys which suffered a change between this and last cycle
        keyDiff = (self.DownKeys ^ actualDownKeys)

        # Just now pressed keys (action to push, not to maintain pushed), are the ones which suffered a change AND actual down keys
        pressedKeys = keyDiff & actualDownKeys

        # Just now released keys are the ones which suffered a change AND previous active keys
        releasedKeys = keyDiff & self.DownKeys

        # Priorities in determining output action: SHOOT | RIGHT | LEFT (Can be combined). Left + Right annulate themselves
        self.InputActions[KEY_INDEX_UP] = (actualDownKeys & POSSIBLE_KEYS[KEY_INDEX_UP][1]) != 0
        self.InputActions[KEY_INDEX_DOWN] = (actualDownKeys & POSSIBLE_KEYS[KEY_INDEX_DOWN][1]) != 0
        self.InputActions[KEY_INDEX_RIGHT] = (actualDownKeys & POSSIBLE_KEYS[KEY_INDEX_RIGHT][1]) != 0
        self.InputActions[KEY_INDEX_LEFT] = (actualDownKeys & POSSIBLE_KEYS[KEY_INDEX_LEFT][1]) != 0
        

        # Update downkeys so in next cycle will be possible to observe press/release changes
        self.DownKeys = actualDownKeys


    def _MoveUp(self):
        self.CrosshairInstance.MoveUp()
        
    def _MoveDown(self):
        self.CrosshairInstance.MoveDown()
    
    def _MoveRight(self):
        self.CrosshairInstance.MoveRight()
                
    def _MoveLeft(self):
        self.CrosshairInstance.MoveLeft()
        
    def _MoveNoneY(self):
        self.CrosshairInstance.MoveNoneY()
        
    def _MoveNoneX(self):
        self.CrosshairInstance.MoveNoneX()
        
    def DiscountBulletTime(self):
        self.NextBulletChangeTimeout -= TIME_PER_CYCLE
        
        if(self.NextBulletChangeTimeout <= 0.0):
            self.NextBulletChangeTimeout = 2.0 + self.Random.random()*3.0
            initialSpeed = (self.Random.random() - 0.5)*2.0*BULLET_SPEED
            initialAngle = self.Random.random()*2.0*math.pi
            [cosx, siny] = [math.cos(initialAngle), math.sin(initialAngle)]
            self.BulletInstance.SetSpeed(pygame.Vector2(initialSpeed*cosx, initialSpeed*siny))

  

    # The most important action, as it processes one game step taking note on given external action and giving an observation for this processed step (Gfx render is not done here)
    def step(self, extAction = ACTION_NONE):
        info = {}

        quited = False
        truncated = False
        

        if(self.Running and (self.render_mode == 'human')):
            # poll for events (this is slow operation, so it is not intended to be done whilst ai training). That will in counterpart freeze game window
            # pygame.QUIT event means the user clicked X to close your window    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.Running = False
                    quited = True
                    break

        if(self.Running):

            #Detect keys if Human is playing, otherwise take external action
            if(self.Mode == GAME_MODE_NORMAL):
                self._KeyDetection()
            else:
                self.InputActions = extAction

            if(self.InputActions[KEY_INDEX_UP] and (not self.InputActions[KEY_INDEX_DOWN])):
                self._MoveUp()
            elif(self.InputActions[KEY_INDEX_DOWN] and (not self.InputActions[KEY_INDEX_UP])):
                self._MoveDown()
            else:
                self._MoveNoneY()

            if(self.InputActions[KEY_INDEX_RIGHT] and (not self.InputActions[KEY_INDEX_LEFT])):
                self._MoveRight()
            elif(self.InputActions[KEY_INDEX_LEFT] and (not self.InputActions[KEY_INDEX_RIGHT])):
                self._MoveLeft()
            else:
                self._MoveNoneX()
                
            self.DiscountBulletTime()

            # Clear virtual output
            self.OutputObs.fill(0)

            # Every-instance-loop (most important part of step)
            for gObject in self.GameObjects:
                gObject.Update()

                virtualwidth = int(gObject.shapeSize.x /OUTPUT_SIZE_FACTOR)
                virtualheight = int(gObject.shapeSize.y /OUTPUT_SIZE_FACTOR)
                virtualposx = int((gObject.position.x - REGION_OF_INTEREST[0]) / OUTPUT_SIZE_FACTOR)
                virtualposy = int((gObject.position.y) / OUTPUT_SIZE_FACTOR)
                        
                virtualxmin = max(0,virtualposx - virtualwidth//2)
                virtualxmax = min(OUTPUT_NP_X_LENGTH-1, virtualposx + virtualwidth//2) + 1
                virtualymin = max(0,virtualposy - virtualheight//2)
                virtualymax = min(OUTPUT_NP_Y_LENGTH-1, virtualposy + virtualheight//2) + 1
    
                if(gObject == self.BulletInstance):
                    self.OutputObs[virtualymin:virtualymax,virtualxmin:virtualxmax,0] = 127
                else:
                    self.OutputObs[virtualymin:virtualymax,virtualxmin:virtualxmax,0] = 255
                
            if(self.CrosshairInstance.rect.colliderect(self.BulletInstance.rect)):
                self.InsideTime += TIME_PER_CYCLE
                

            # Increment elapsed time
            self.ElapsedTime +=TIME_PER_CYCLE
            self.OutsideTime = self.ElapsedTime - self.InsideTime

            # End game when time is over
            if((self.OutsideTime >= MAX_TIME_OUTSIDE) or (self.ElapsedTime >= ROUND_TIME_S)):
                self.Running = False
                truncated = True
        

         
        info['BulletPosition'] = [self.BulletInstance.position.x, self.BulletInstance.position.y]
        info['CrosshairPosition'] = [self.CrosshairInstance.position.x, self.CrosshairInstance.position.y]
        info['DeltaPosition'] = [self.BulletInstance.position.x - self.CrosshairInstance.position.x,\
            self.BulletInstance.position.y - self.CrosshairInstance.position.y]
        info['ElapsedTime'] = self.ElapsedTime
        info['InsideTime'] = self.InsideTime
        info['OutsideTime'] = self.OutsideTime
        

        if(quited):
            truncated = True
            self.close()
            
        done = not self.Running and not truncated
                
        return self.OutputObs, done, truncated, info

    # Rendering is optional when training AI, but is required in Env wrapper
    def render(self):
        if(self.Running):
            # fill the screen with a color to wipe away anything from last frame
            self.Screen.fill('black')

            # Render objects with their own custom function
            for gObject in self.GameObjects:
                gObject.Draw(self.Screen)

            # Info text
            img = self.Font.render('SCORE: '+str(self.Score), True, 'white')
            self.Screen.blit(img, (10, 10))
            img = self.Font.render('ELAPSED TIME: '+str(int(self.ElapsedTime)), True, 'white')
            self.Screen.blit(img, (10, 30))
            img = self.Font.render('INSIDE TIME: '+str(int(self.InsideTime)), True, 'white')
            self.Screen.blit(img, (10, 50))
            img = self.Font.render('OUTSIDE TIME: '+str(int(self.OutsideTime)), True, 'white')
            self.Screen.blit(img, (10, 70))
        
            # flip() the display to put your work on screen
            pygame.display.flip()

            # limits FPS to 60 when playing rendered
            self.Clock.tick(EXPECTED_FPS)  





