import random
import numpy as np
from paratroopergame_objects import *
from paratroopergame_cfg import *
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
            
            # This is a fast-access to Paratroopers, the first one will always be the one which is nearest to ground
            self.ActiveParatroopers = []

            # Declare empty pools
            self.PooledAircrafts = []
            self.PooledParatroopers = []
            self.PooledBullets = []
            self.PooledExplosions = []

            # Initialize action box to all 0
            self.InputActions = np.zeros(len(POSSIBLE_KEYS))

            # Own Random instance (for seeding purposes and in order to not to interact with global library)
            self.Random = random.Random(seed)

            # Pass GameObjects list to class static variable (this make possible automatic register/de-register when creating or destroying)
            GameObject.GameObjects = self.GameObjects
            GameObject.Random = self.Random

            # Pass Active Bullets list to class static variable (this make possible automatic register/de-register when creating or destroying)
            GameObject.ActiveBullets = self.ActiveBullets
            GameObject.ActiveParatroopers = self.ActiveParatroopers
            GameObject.PooledBullets = self.PooledBullets
            GameObject.PooledAircrafts = self.PooledAircrafts
            GameObject.PooledParatroopers = self.PooledParatroopers
            GameObject.PooledExplosions = self.PooledExplosions

            # Prepare Pools. This strategy allocates game instances memory, so it will be necessary no more to create instances, just reuse them
            for _ in range(POOL_SIZE):
                self.PooledBullets.append(Bullet())
                self.PooledExplosions.append(Explosion())
                self.PooledAircrafts.append(Aircraft())
                self.PooledParatroopers.append(Paratrooper())


            # Create a Wall under Cannon
            wallHeight = SCREEN_SIZE[1]-CANNON_HEIGHT-CANNON_SEGMENT_RADIUS_HALF
            self.WallInstance = Wall()
            self.WallInstance.ReCreate(pygame.Vector2(SCREEN_SIZE[0]//2, SCREEN_SIZE[1] - wallHeight//2), pygame.Vector2(128, wallHeight))

            # Create Cannon instance
            self.CannonInstance = Cannon()
            self.CannonInstance.ReCreate(pygame.Vector2(SCREEN_SIZE[0]//2, CANNON_HEIGHT))
                

            # Prepare elapsed time
            self.ElapsedTime = 0.0

            #Create output observation array
            self.OutputObs = np.zeros((OUTPUT_NP_Y_LENGTH, OUTPUT_NP_X_LENGTH, 1))

            # Running game = True (by the moment)
            self.Running = True

            # Score initialization
            self.Score = 0

            # Number of paratroopers which reached bottom
            self.ParatroopersReached = 0
            self.ParatroopersDestroyed = 0
            self.AircraftsDestroyed = 0
            self.MissedBullets = 0

            # Bullet initial values
            self.BulletReady = True
            self.ReloadTimeout = 0

            

            # Aircraft spawn timeout
            self.AircraftTimeout = self.Random.randint(AIRCRAFT_SPAWN_TIME[0],AIRCRAFT_SPAWN_TIME[1])

            # Neutral pressed keys in bit field is 0
            self.DownKeys = 0x0

            # Launch a first black screen to window
            if(self.Screen != None):
                self.Screen.fill('black')
                pygame.display.flip()

            return self.OutputObs
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
        self.InputActions[KEY_INDEX_JUMP] = pressedKeys & POSSIBLE_KEYS[KEY_INDEX_JUMP][1] != 0
        self.InputActions[KEY_INDEX_RIGHT] = actualDownKeys & POSSIBLE_KEYS[KEY_INDEX_RIGHT][1] != 0
        self.InputActions[KEY_INDEX_LEFT] = actualDownKeys & POSSIBLE_KEYS[KEY_INDEX_LEFT][1] != 0

        # Update downkeys so in next cycle will be possible to observe press/release changes
        self.DownKeys = actualDownKeys


    def _Shoot(self):
        if(self.BulletReady):
            if(len(self.PooledBullets)>0):
                newBullet = self.PooledBullets.pop(0)

                speedx = BULLET_SPEED_ADVANCE_PER_CYCLE * self.CannonInstance.cos_angle
                speedy = -BULLET_SPEED_ADVANCE_PER_CYCLE * self.CannonInstance.sin_angle

                newBullet.ReCreate(pygame.Vector2(self.CannonInstance.position), pygame.Vector2(speedx, speedy))
    
                self.BulletReady = False
                self.ReloadTimeout = BULLET_RELOAD_TIME_CYCLES
            else:
                raise Exception('Critical Error, bullet pool exhausted')
                self.Running = False

    def _ShootReload(self):
        if(self.ReloadTimeout > 0):
            self.ReloadTimeout -= 1
            if(self.ReloadTimeout == 0):
                self.BulletReady = True

    
    def _MoveRight(self):
        self.CannonInstance.MoveRight()
                
    def _MoveLeft(self):
        self.CannonInstance.MoveLeft()

    def _AircraftSpawn(self):

        #Subtract 1 cycle to timeout
        self.AircraftTimeout -= 1

        #In case it reaches 0, spawn a new aircraft and reload timer. Timer must always be reloaded.
        if(self.AircraftTimeout == 0):
            if(len(self.PooledAircrafts) > 0):
                #Decide its direction and speed
                aircraft_speed = self.Random.randint(AIRCRAFT_SPEED_RANGE[0], AIRCRAFT_SPEED_RANGE[1])
                aircraft_dir = self.Random.choice([-1,1])

                if(aircraft_dir == 1):
                    spawn_pos = pygame.Vector2(0, 30)
                else:
                    spawn_pos = pygame.Vector2(SCREEN_SIZE[0], 30)

                #Take first of pool and recreate it
                aircraft = self.PooledAircrafts.pop(0)
                aircraft.ReCreate(spawn_pos, pygame.Vector2(aircraft_speed*aircraft_dir, 0))
            else:
                raise Exception('Pool of aircrafts exhausted. Fatal error')
                self.Running = False

            #Reload timeout
            self.AircraftTimeout = self.Random.randint(AIRCRAFT_SPAWN_TIME[0], AIRCRAFT_SPAWN_TIME[1])

    

  

    # The most important action, as it processes one game step taking note on given external action and giving an observation for this processed step (Gfx render is not done here)
    def step(self, extAction = ACTION_NONE):
        info = {}

        quited = False
        truncated = False
        
        lowestParatrooperExists = False
        #Only draw lowest paratrooper
        lowestParatrooperPosition = [0,0]


        if(self.Running and (self.render_mode == 'human')):
            # poll for events (this is slow operation, so it is not intended to be done whilst ai training). That will in counterpart freeze game window
            # pygame.QUIT event means the user clicked X to close your window    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.Running = False
                    quited = True
                    break

        if(self.Running):

            #Manage shoot reload time
            self._ShootReload()

            #Manage Aircraft spawn
            self._AircraftSpawn()

            #Detect keys if Human is playing, otherwise take external action
            if(self.Mode == GAME_MODE_NORMAL):
                self._KeyDetection()
            else:
                self.InputActions = extAction

            #Manage given actions
            if(self.InputActions[KEY_INDEX_JUMP]):
                self._Shoot()

            if(self.InputActions[KEY_INDEX_RIGHT] and (not self.InputActions[KEY_INDEX_LEFT])):
                self._MoveRight()

            if(self.InputActions[KEY_INDEX_LEFT] and (not self.InputActions[KEY_INDEX_RIGHT])):
                self._MoveLeft()

            # Clear virtual output
            self.OutputObs.fill(0)

            # Every-instance-loop (most important part of step)
            for gObject in self.GameObjects:
                gObject.Update()

                if(gObject.killed != KILLED_NOT):
                    if(gObject.objType == OBJ_TYPE_PARATROOPER):
                        if(gObject.killed == KILLED_WASTED):
                            self.ParatroopersReached += 1
                            self.Score -= 10
                        else:
                            self.ParatroopersDestroyed += 1
                            self.Score += 1
                    elif(gObject.objType == OBJ_TYPE_BULLET):
                        if(gObject.killed == KILLED_WASTED):
                            self.MissedBullets += 1
                    elif(gObject.objType == OBJ_TYPE_AIRCRAFT):
                        if(gObject.killed == KILLED_BINGO):
                            self.AircraftsDestroyed += 1
                            self.Score += 10

                    gObject.Destroy()

            
            
            if(len(self.ActiveParatroopers) > 0):
                gObject = self.ActiveParatroopers[0]
                
                lowestParatrooperExists = True
                lowestParatrooperPosition[0] = gObject.position.x
                lowestParatrooperPosition[1] = gObject.position.y
                
                if((gObject.position.x > REGION_OF_INTEREST[0])and(gObject.position.x < REGION_OF_INTEREST[1])):
                    virtualwidth = int(gObject.shapeSize.x /OUTPUT_SIZE_FACTOR)
                    virtualheight = int(gObject.shapeSize.y /OUTPUT_SIZE_FACTOR)
                    virtualposx = int((gObject.position.x - REGION_OF_INTEREST[0]) / OUTPUT_SIZE_FACTOR)
                    virtualposy = int((gObject.position.y) / OUTPUT_SIZE_FACTOR)
                            
                    virtualxmin = max(0,virtualposx - virtualwidth//2)
                    virtualxmax = min(OUTPUT_NP_X_LENGTH-1, virtualposx + virtualwidth//2) + 1
                    virtualymin = max(0,virtualposy - virtualheight//2)
                    virtualymax = min(OUTPUT_NP_Y_LENGTH-1, virtualposy + virtualheight//2) + 1

                    self.OutputObs[virtualymin:virtualymax,virtualxmin:virtualxmax,0] = 255
                
        
            
            # Prepare output of cannon aim (Laser)
            if(False):
                laserbasex = self.CannonInstance.position.x / OUTPUT_SIZE_FACTOR
                laserbasey = self.CannonInstance.position.y / OUTPUT_SIZE_FACTOR
                for i in range(LASER_SEGMENTS):
                    laserx = int(laserbasex + self.CannonInstance.cos_angle * RADIAL_LASER_DISTANCE * i)
                    laserx = max(0,laserx)
                    laserx = min(OUTPUT_NP_X_LENGTH-1, laserx)
                    lasery = int(laserbasey - self.CannonInstance.sin_angle * RADIAL_LASER_DISTANCE * i)
                    lasery = max(0,lasery)
                    lasery = min(OUTPUT_NP_Y_LENGTH-1, lasery)
                    self.OutputObs[lasery,laserx,0] = 127

            # Increment elapsed time
            self.ElapsedTime +=TIME_PER_CYCLE
   

            # End game when time is over
            if(self.ParatroopersReached >= MAX_PARATROOPERS_REACH_BOTTOM):
                self.Running = False
                truncated = True
        

         
        info['CannonAngleCosSin'] = [self.CannonInstance.cos_angle, self.CannonInstance.sin_angle]
        info['CannonPosition'] = [self.CannonInstance.position.x, self.CannonInstance.position.y]
        info['LowestParatrooperExists'] = lowestParatrooperExists
        info['LowestParatrooperPosition'] = lowestParatrooperPosition
        info['DestroyedParatroopers'] = self.ParatroopersDestroyed
        info['EscapedParatroopers'] = self.ParatroopersReached
        info['MissedBullets'] = self.MissedBullets
        info['DestroyedAircrafts'] = self.AircraftsDestroyed
        info['ElapsedTime'] = self.ElapsedTime
        

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
            img = self.Font.render('ESCAPED PARATROOPERS: '+str(self.ParatroopersReached)+' / '+str(MAX_PARATROOPERS_REACH_BOTTOM), True, 'white')
            self.Screen.blit(img, (10, 30))
            img = self.Font.render('DESTROYED PARATROOPERS: '+str(self.ParatroopersDestroyed), True, 'white')
            self.Screen.blit(img, (10, 50))
            img = self.Font.render('DESTROYED AIRCRAFTS: '+str(self.AircraftsDestroyed), True, 'white')
            self.Screen.blit(img, (10, 70))
            img = self.Font.render('MISSED BULLETS: '+str(self.MissedBullets), True, 'white')
            self.Screen.blit(img, (10, 90))
            img = self.Font.render('ELAPSED TIME: '+str(int(self.ElapsedTime)), True, 'white')
            self.Screen.blit(img, (10, 110))
        
            # flip() the display to put your work on screen
            pygame.display.flip()

            # limits FPS to 60 when playing rendered
            self.Clock.tick(EXPECTED_FPS)  





