from paratroopergame_cfg import *
import pygame
import math



class GameObject:

    #Static Variable (same for ALL objects). Game System will pass its value on startup
    GameObjects = None
    ActiveBullets = None
    ActiveParatroopers = None
    PooledBullets = None
    PooledAircrafts = None
    PooledParatroopers = None
    PooledExplosions = None
    Random = None

    def __init__(self, objType):
        self.objType = objType 

    def _reset(self,shapeType, shapeColor, shapeSize):
        self.shapeType = shapeType
        self.shapeColor = shapeColor
        self.shapeSize = shapeSize
        self.position = pygame.Vector2(0, 0)
        self.speed = pygame.Vector2(0, 0)
        self.rect = pygame.Rect(self.position.x, self.position.y, self.shapeSize.x, self.shapeSize.y)
        self.killed = KILLED_NOT
        

    def ReCreate(self, shapeType, shapeColor, shapeSize):
        self._reset(shapeType,shapeColor,shapeSize)
        GameObject.GameObjects.append(self)

    def Destroy(self):
        GameObject.GameObjects.remove(self)
        
    def Update(self):
        self.position.x += self.speed.x
        self.position.y += self.speed.y

        self.rect.width = self.shapeSize.x
        self.rect.height = self.shapeSize.y
        
        self.rect.left = self.position.x - self.rect.width*0.5
        self.rect.top = self.position.y - self.rect.height*0.5

    def Kill(self, reason):
        self.killed = reason 

    def Draw(self, surface):
        if(self.shapeType == SHAPE_CIRCLE):
            pygame.draw.circle(surface, self.shapeColor, self.position, self.shapeSize.x)
        else:
            pygame.draw.rect(surface, self.shapeColor, self.rect)

    @classmethod
    def CreateExplosion(cls, position):
        if(len(cls.PooledExplosions) > 0):
            newExplosion = cls.PooledExplosions.pop(0)
            newExplosion.ReCreate(pygame.Vector2(position))
        else:
            raise Exception('Pool of explosions is exhausted. Fatal error')



class Cannon(GameObject):
    def __init__(self):
        super().__init__(OBJ_TYPE_CANNON)
        self.segments = []
        for _ in range(CANNON_SEGMENTS):
            self.segments.append(pygame.Rect(0, 0, CANNON_SEGMENT_SIZE[0], CANNON_SEGMENT_SIZE[1]))

    def ReCreate(self, position):
        super().ReCreate(SHAPE_RECTANGLE, 'white', pygame.Vector2(CANNON_SIZE))

        self.position = position
        self.angle = (CANNON_MAX_ANGLE + CANNON_MIN_ANGLE) / 2.0

        self.cos_angle = math.cos(self.angle)
        self.sin_angle = math.sin(self.angle)

        self.segments[0].centerx = position.x
        self.segments[0].centery = position.y

    def MoveRight(self):
        self.angle = max(CANNON_MIN_ANGLE, self.angle - CANNON_ANGLE_ADVANCE)
        self.cos_angle = math.cos(self.angle)
        self.sin_angle = math.sin(self.angle)

    def MoveLeft(self):
        self.angle = min(CANNON_MAX_ANGLE, self.angle + CANNON_ANGLE_ADVANCE)
        self.cos_angle = math.cos(self.angle)
        self.sin_angle = math.sin(self.angle)

    def Update(self):
        super().Update()

    #Override parent class (don't let it draw standard rectangle), super() won't be called
    def Draw(self, surface):
        pygame.draw.rect(surface, self.shapeColor,self.segments[0])

        for i in range(1,CANNON_SEGMENTS):
            radius = CANNON_SEGMENT_RADIUS*(i)
            self.segments[i].centerx = self.position.x + radius*self.cos_angle
            self.segments[i].centery = self.position.y - radius*self.sin_angle
            pygame.draw.rect(surface, self.shapeColor,self.segments[i])

class Bullet(GameObject):
    def __init__(self):
        super().__init__(OBJ_TYPE_BULLET)

    def ReCreate(self, position, initialSpeed):
        super().ReCreate(SHAPE_CIRCLE, 'red', pygame.Vector2(4,4))

        self.WasUseful = False
        self.position = position
        self.speed = initialSpeed

        GameObject.ActiveBullets.append(self)

    def Destroy(self):
        GameObject.ActiveBullets.remove(self)
        GameObject.PooledBullets.append(self)
        super().Destroy()

    def Update(self):
        super().Update()
        if(not SCREEN_RECT.collidepoint(self.position)):
            if(self.WasUseful):
                self.Kill(KILLED_NEUTRAL)
            else:
                self.Kill(KILLED_WASTED)

    def setWasUseful(self):
        self.WasUseful = True


class Explosion(GameObject):
    def __init__(self):
        super().__init__(OBJ_TYPE_EXPLOSION)

    def ReCreate(self, position):
        super().ReCreate(SHAPE_CIRCLE, 'yellow', pygame.Vector2(2,2))

        self.Timeout = 30
        self.position = position

    def Destroy(self):
        GameObject.PooledExplosions.append(self)
        super().Destroy()

    def Update(self):
        super().Update()
        self.Timeout -= 1
        if(self.Timeout <= 0):
            self.Kill(KILLED_NEUTRAL)
        else:
            radius = (30 - self.Timeout) * 10 / 30
            self.shapeSize = pygame.Vector2(radius,radius)


class Aircraft(GameObject):
    def __init__(self):
        super().__init__(OBJ_TYPE_AIRCRAFT)

    def ReCreate(self, position, speed):
        super().ReCreate(SHAPE_RECTANGLE, 'blue', pygame.Vector2(48,16))

        self.position = position
        self.speed = speed

        droparea = GameObject.Random.choice([0,1])
        self.dropX = GameObject.Random.randint(DROP_ZONES[droparea][0], DROP_ZONES[droparea][1])
        self.dropped = False

    def Destroy(self):
        GameObject.PooledAircrafts.append(self)
        super().Destroy()

    def Update(self):
        super().Update()
        if(((self.speed.x > 0) and (self.position.x > SCREEN_SIZE[0]))or((self.speed.x) < 0 and (self.position.x < 0))):
            self.Kill(KILLED_WASTED)
        else:
            if((not self.dropped)and(((self.speed.x < 0) and (self.dropX >= self.position.x))or((self.speed.x > 0) and (self.dropX <= self.position.x)))):
                self.dropped = True
                if(len(GameObject.PooledParatroopers)>0):
                    paratrooper = GameObject.PooledParatroopers.pop(0)
                    paratrooper.ReCreate(pygame.Vector2(self.dropX, self.position.y), pygame.Vector2(0,PARATROOPER_FALL_SPEED))
                else:
                    raise Exception('Pool of paratroopers exhausted, critical error')
            #Check collision against all active bullets
            for bullet in GameObject.ActiveBullets:
                if(self.rect.colliderect(bullet.rect)):
                    bullet.setWasUseful()
                    self.Kill(KILLED_BINGO)
                    GameObject.CreateExplosion(bullet.position)
                    break


class Paratrooper(GameObject):
    def __init__(self):
        super().__init__(OBJ_TYPE_PARATROOPER)

    def ReCreate(self, position, speed):
        super().ReCreate(SHAPE_RECTANGLE, 'orange', pygame.Vector2(32,32))

        self.position = position
        self.speed = speed

        GameObject.ActiveParatroopers.append(self)

    def Destroy(self):
        GameObject.ActiveParatroopers.remove(self)
        GameObject.PooledParatroopers.append(self)
        super().Destroy()

    def Update(self):
        super().Update()

        if(self.position.y > SCREEN_SIZE[1]):
            self.Kill(KILLED_WASTED)
        else:
            for bullet in GameObject.ActiveBullets:
                if(self.rect.colliderect(bullet.rect)):
                    bullet.setWasUseful()
                    self.Kill(KILLED_BINGO)
                    GameObject.CreateExplosion(bullet.position)
                    break
        


#Decorative
class Wall(GameObject):
    def __init__(self):
        super().__init__(OBJ_TYPE_WALL)

    def ReCreate(self, position, size):
        super().ReCreate(SHAPE_RECTANGLE, 'gray', size)

        self.position = position