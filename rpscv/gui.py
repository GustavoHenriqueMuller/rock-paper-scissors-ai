import sys
import numpy as np

import cv2
import pygame as pg
import pygame.freetype

from rpscv import utils

class RPSGUI():
    def __init__(self, loop = False):
        pg.init()

        self.loop = loop
        self.sWidth = 640
        self.sHeight = 480
        self.surf = pg.display.set_mode((self.sWidth, self.sHeight))

        pg.display.set_caption('Rock paper scissors - Trabalho M3 IA')

        self.playerScore = 0
        self.computerScore = 0
        self.playerImage = pg.Surface((200, 300))
        self.computerImage = pg.Surface((200, 300))
        self.playerImagePosition = (60, 160)
        self.computerImagePosition = (380, 160)
        self.playerZone = pg.Surface((250, 330))
        self.computerZone = pg.Surface((250, 330))
        self.playerZonePosition = (35, 145)
        self.computerZonePosition = (355, 145)
        self.winner = None

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

    def blitTextAlignCenter(self, surf, text, pos):
        tWidth = text[1].width
        surf.blit(text[0], (pos[0] - tWidth / 2, pos[1]))

    def draw(self):
        # Fill surface with background color
        self.surf.fill(self.WHITE)

        # Draw boxes around computer and player areas
        playerVertices = [(5, 3), (315, 3), (315, 476), (5, 476), (5, 3)]
        pg.draw.polygon(self.surf, self.BLACK, playerVertices, 1)

        computerVertices = [(325, 3), (634, 3), (634, 476), (325, 476), (325, 3)]
        pg.draw.polygon(self.surf, self.BLACK, computerVertices, 1)

        # Render computer and player text
        font = pg.freetype.SysFont(None, 30)
        text = font.render('PLAYER', self.BLACK)
        self.blitTextAlignCenter(self.surf, text, (160,15))
        text = font.render('COMPUTER', self.BLACK)
        self.blitTextAlignCenter(self.surf, text, (480,15))

        # Set computer and player zone colors
        if self.winner == 'player':
            self.playerZone.fill(self.GREEN)
            self.computerZone.fill(self.RED)
        elif self.winner == 'computer':
            self.playerZone.fill(self.RED)
            self.computerZone.fill(self.GREEN)
        elif self.winner == 'tie':
            self.playerZone.fill(self.BLUE)
            self.computerZone.fill(self.BLUE)
        else:
            self.playerZone.fill(self.WHITE)
            self.computerZone.fill(self.WHITE)

        # Blit computer and player zone
        self.surf.blit(self.playerZone, self.playerZonePosition)
        self.surf.blit(self.computerZone, self.computerZonePosition)

        # Blit computer and player images
        self.surf.blit(self.playerImage, self.playerImagePosition)
        self.surf.blit(self.computerImage, self.computerImagePosition)

        # Render computer and player scores
        font = pg.freetype.SysFont(None, 100)
        text = font.render(str(self.playerScore), self.BLACK)
        self.blitTextAlignCenter(self.surf, text, (160, 60))
        text = font.render(str(self.computerScore), self.BLACK)
        self.blitTextAlignCenter(self.surf, text, (480, 60))

    def gameOver(self, delay = 3500):
        # Create surface for Game Over message
        goZone = pg.Surface((400, 200))

        # Fill surface with background color
        goZone.fill(self.WHITE)

        # Draw box around surface
        vertices = [(3, 3), (396, 3), (396, 196), (3, 196), (3, 3)]
        pg.draw.polygon(goZone, self.BLACK, vertices, 1)

        # Render text on surface
        font = pg.freetype.SysFont(None, 40)
        gameOverText = font.render('GAME OVER', self.BLACK)
        self.blitTextAlignCenter(goZone, gameOverText, (200, 45))

        if self.playerScore > self.computerScore:
            winner = 'PLAYER'
            color = self.GREEN
        else:
            winner = 'COMPUTER'
            color = self.RED

        winnerText = font.render('{} WINS!'.format(winner), color)
        self.blitTextAlignCenter(goZone, winnerText, (200, 110))

        # Blit goZone to main surface
        pos = (self.sWidth / 2 - 200, 175)
        self.surf.blit(goZone, pos)

        pg.display.flip()
        pg.time.wait(delay)

        if self.loop:
            self.reset()
        else:
            self.quit()

    def quit(self, delay = 0):
        pg.time.wait(delay)
        pg.quit()
        sys.exit()

    def reset(self):
        self.playerScore = 0
        self.computerScore = 0

    def setPlayerMove(self, image, gestureIndex):
        font = pg.freetype.SysFont(None, 40)

        playerMoveZone = pg.Surface((400, 200))
        playerMoveText = font.render(utils.getGestureNameByIndex(gestureIndex), self.BLACK)

        self.blitTextAlignCenter(playerMoveZone, playerMoveText, (200, 110))
        self.playerImage = pg.surfarray.make_surface(image[::-1,:,:])
    
    def setComputerMove(self, image, gestureIndex):
        self.computerImage = pg.surfarray.make_surface(image[:,::-1,:])   

    def setWinner(self, winner):
        self.winner = winner

        if winner == 'player':
            self.playerScore += 1
        elif winner == 'computer':
            self.computerScore += 1

        
