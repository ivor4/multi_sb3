from crosshairgame_cfg import *
import pygame
import math



class GameObject:

    #Static Variable (same for ALL objects). Game System will pass its value on startup
    GameObjects = None
    ActiveBullets = None
    ActiveCrosshairs = None
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
        self.position.x = min(max(0, self.position.x + self.speed.x), SCREEN_SIZE[0])
        self.position.y = min(max(0, self.position.y + self.speed.y), SCREEN_SIZE[1])

        self.rect.width = self.shapeSize.x
        self.rect.height = self.shapeSize.y
        
        self.rect.left = self.position.x - self.rect.width*0.5
        self.rect.top = self.position.y - self.rect.height*0.5


    def Draw(self, surface):
        if(self.shapeType == SHAPE_CIRCLE):
            pygame.draw.circle(surface, self.shapeColor, self.position, self.shapeSize.x)
        else:
            pygame.draw.rect(surface, self.shapeColor, self.rect)




class CrossHair(GameObject):
    def __init__(self):
        super().__init__(OBJ_TYPE_CROSSHAIR)

    def ReCreate(self, position):
        super().ReCreate(SHAPE_CIRCLE, 'white', pygame.Vector2(CROSSHAIR_SIZE))
        self.position = position
        self.speed.x = 0
        self.speed.y = 0
        
        GameObject.ActiveCrosshairs.append(self)

    def MoveUp(self):
        self.speed.y = -CROSSHAIR_SPEED
        
    def MoveDown(self):
        self.speed.y = CROSSHAIR_SPEED

    def MoveRight(self):
        self.speed.x = CROSSHAIR_SPEED

    def MoveLeft(self):
        self.speed.x = -CROSSHAIR_SPEED
        
    def MoveNoneX(self):
        self.speed.x = 0
        
    def MoveNoneY(self):
        self.speed.y = 0



class Bullet(GameObject):
    def __init__(self):
        super().__init__(OBJ_TYPE_BULLET)

    def ReCreate(self, position, initialSpeed):
        super().ReCreate(SHAPE_RECTANGLE, 'gray', pygame.Vector2(CROSSHAIR_SIZE[0]*4, CROSSHAIR_SIZE[1]*4))

        self.position = position
        self.speed = initialSpeed

        GameObject.ActiveBullets.append(self)
        
    def SetSpeed(self, speed):
        self.speed = speed


