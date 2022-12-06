import sys
import numpy as np

import cv2
import pygame as pg
import pygame.freetype

from rpscv import utils

class RPSGUI():
    def __init__(self):
        pg.init()

        self.loop = True
        self.sWidth = 640
        self.sHeight = 480
        self.surf = pg.display.set_mode((self.sWidth, self.sHeight))

        pg.display.set_caption('Rock paper scissors - Trabalho M3 IA')

        self.playerScore = 0
        self.computerScore = 0
        self.playerMove = -1
        self.computerMove = -1
        self.playerImage = pg.Surface((200, 300))
        self.computerImage = pg.Surface((200, 300))
        self.playerImagePosition = (60, 160)
        self.computerImagePosition = (380, 160)
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
        self.surf.fill(self.BLACK)

        # Render computer and player text
        font = pg.freetype.SysFont(None, 30)

        text = font.render('PLAYER', self.WHITE)
        self.blitTextAlignCenter(self.surf, text, (160, 105))

        if self.playerMove != -1:
            font = pg.freetype.SysFont(None, 20)
            text = font.render(utils.gestureTexts[self.playerMove].capitalize(), self.WHITE)
            self.blitTextAlignCenter(self.surf, text, (160, 135))

        font = pg.freetype.SysFont(None, 30)
        text = font.render('COMPUTER', self.WHITE)
        self.blitTextAlignCenter(self.surf, text, (480, 105))

        if self.computerMove != -1:
            font = pg.freetype.SysFont(None, 20)
            text = font.render(utils.gestureTexts[self.computerMove].capitalize(), self.WHITE)
            self.blitTextAlignCenter(self.surf, text, (480, 135))

        font = pg.freetype.SysFont(None, 30)

        # Blit computer and player images
        self.surf.blit(self.playerImage, self.playerImagePosition)
        self.surf.blit(self.computerImage, self.computerImagePosition)

        # Render computer and player scores
        font = pg.freetype.SysFont(None, 80)
        text = font.render(str(self.playerScore), self.WHITE)
        self.blitTextAlignCenter(self.surf, text, (160, 15))
        text = font.render(str(self.computerScore), self.WHITE)
        self.blitTextAlignCenter(self.surf, text, (480, 15))

    def gameOver(self, delay = 3500):
        # Create surface for Game Over message
        goZone = pg.Surface((400, 200))

        # Fill surface with background color
        goZone.fill(self.BLACK)

        # Draw box around surface
        vertices = [(3, 3), (396, 3), (396, 196), (3, 196), (3, 3)]
        pg.draw.polygon(goZone, self.BLACK, vertices, 1)

        # Render text on surface
        font = pg.freetype.SysFont(None, 40)
        gameOverText = font.render('GAME OVER', self.WHITE)
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
        playerMoveText = font.render(utils.gestureTexts[gestureIndex], self.BLACK)

        self.blitTextAlignCenter(playerMoveZone, playerMoveText, (200, 110))
        self.playerMove = gestureIndex
        self.playerImage = pg.surfarray.make_surface(image[::-1,:,:])
    
    def setComputerMove(self, image, gestureIndex):
        self.computerMove = gestureIndex
        self.computerImage = pg.surfarray.make_surface(image[:,::-1,:])   

    def setWinner(self, winner):
        self.winner = winner

        if winner == 'player':
            self.playerScore += 1
        elif winner == 'computer':
            self.computerScore += 1

        
