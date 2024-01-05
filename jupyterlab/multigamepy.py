import pygame

class MultiGameManager:
    def __init__(self):
        self.instances = []

        if(not pygame.get_init()):
            pygame.init()
            self.OK = True
        else:
            self.OK = False
            raise Exception("Pygame already running")

    def close(self):
        for game in self.instances:
            game.close()

        if(pygame.get_init()):
            pygame.quit()

        while(pygame.get_init()):
            pass

    def regGame(self, game):
        self.instances.append(game)

    def unregGame(self, game):
        self.instances.remove(game)