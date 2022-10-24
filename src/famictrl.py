# -*- coding: utf-8 -*-
''' Transfer required libraries '''
import pygame
import sys
import random
from math import *
from random import randint

import time

''' Generate random colors '''
white = (230, 230, 230)
lightBlue = (4, 27, 96)
red = (231, 76, 60)
lightGreen = (25, 111, 61)
darkGray = (40, 55, 71)
darkBlue = (64, 178, 239)
green = (35, 155, 86)
yellow = (244, 208, 63)
blue = (46, 134, 193)
purple = (155, 89, 182)
orange = (243, 156, 18)

''' Check the balloons '''
def isonBalloon(x, y, a, b, pos):
    if (x < pos[0] < x + a) and (y < pos[1] < y + b):
        return True
    else:
        return False


class Bar():
    def __init__(self, screen, rect, font,  bar = blue, outline = darkGray):
        self.rect = pygame.Rect(rect)
        self.bar = bar
        self.outline = outline
        self.value = 0
        self.font = font
        self.screen = screen
    def draw(self, surf, title):
        length = round(self.value * self.rect.height / 100)
        top = self.rect.height - length
        if self.value > 50:
            pygame.draw.rect(surf, orange, (self.rect.x, self.rect.y + top, self.rect.width, length))
        else:
            pygame.draw.rect(surf, blue, (self.rect.x, self.rect.y + top, self.rect.width, length))
        #pygame.draw.rect(surf, self.outline, self.rect, 2) 
        txt = self.font.render(title, True, darkGray)
        txt_rect = txt.get_rect(topright = (self.rect.x + 30, self.rect.y - 30))
        self.screen.blit(txt, txt_rect)


''' Create a class to do all balloon related operations '''
class Balloon:
    ''' Specify properties of balloons in start function '''
    def __init__(self, speed, display, margin, width, height, lowerBound):
        self.display = display
        self.margin = margin
        self.width = width
        self.height = height
        self.lowerBound = lowerBound
        self.a = random.randint(30, 40)
        self.b = self.a + random.randint(0, 10)
        self.x = random.randrange(self.margin, self.width - self.a - self.margin)
        self.y = self.height - self.lowerBound
        self.angle = 90
        self.speed = -speed
        self.proPool= [-1, -1, -1, 0, 0, 0, 0, 1, 1, 1]
        self.length = random.randint(50, 100)
        self.color = random.choice([red, green, purple, orange, yellow, blue])
        
    ''' Animate balloons using mathematical operators '''
    def move(self):
        direct = random.choice(self.proPool)

        if direct == -1:
            self.angle += -10
        elif direct == 0:
            self.angle += 0
        else:
            self.angle += 10

        self.y += self.speed*sin(radians(self.angle))
        self.x += self.speed*cos(radians(self.angle))

        if (self.x + self.a > self.width) or (self.x < 0):
            if self.y > self.height/5:
                self.x -= self.speed*cos(radians(self.angle)) 
            else:
                self.reset()
        if self.y + self.b < 0 or self.y > self.height + 30:
            self.reset()
            
    ''' Show balloons on screen '''
    def show(self):
        pygame.draw.line(self.display, darkBlue, (self.x + self.a/2, self.y + self.b), (self.x + self.a/2, self.y + self.b + self.length))
        pygame.draw.ellipse(self.display, self.color, (self.x, self.y, self.a, self.b))
        pygame.draw.ellipse(self.display, self.color, (self.x + self.a/2 - 5, self.y + self.b - 3, 10, 10))
            
    ''' Destroy by mouse click on the balloon '''
    def burst(self, score):
        #global score
        pos = pygame.mouse.get_pos()

        if isonBalloon(self.x, self.y, self.a, self.b, pos):
            score += 1
            self.reset()
        return score
    ''' The process of resetting the balloons '''            
    def reset(self):
        self.a = random.randint(30, 40)
        self.b = self.a + random.randint(0, 10)
        self.x = random.randrange(self.margin, self.width - self.a - self.margin)
        self.y = self.height - self.lowerBound 
        self.angle = 90
        self.speed -= 0.002
        self.proPool = [-1, -1, -1, 0, 0, 0, 0, 1, 1, 1]
        self.length = random.randint(50, 100)
        self.color = random.choice([red, green, purple, orange, yellow, blue])


        



class Game:
    def __init__(self, events):
        ''' Create A Desktop Window '''
        pygame.init()
        self.width = 706
        self.height = 386
        self.display = pygame.display.set_mode((self.width, self.height))
        self.bg_neutral = pygame.image.load("ctrl_neutral.png")
        self.bg_up = pygame.image.load("ctrl_up.png")
        self.bg_down = pygame.image.load("ctrl_down.png")
        self.bg_left = pygame.image.load("ctrl_left.png")
        self.bg_right = pygame.image.load("ctrl_right.png")
        self.bg_B = pygame.image.load("ctrl_B.png")
        self.bg_A = pygame.image.load("ctrl_A.png")
        self.bg = self.bg_neutral
        self.events = events
        self.E_NEUTRAL, self.E_UP, self.E_DOWN, self.E_LEFT, self.E_RIGHT, self.E_B, self.E_A = events

        #self.display.blit(self.bg, (0, 0))
        pygame.display.set_caption("Reservoir Computing - the cardboard controller")
        self.clock = pygame.time.Clock()

        ''' Create variable for drawing and score '''
        self.margin = 100
        self.lowerBound = 100
        self.score = 0



        ''' Set the general font of the project '''
        self.font = pygame.font.SysFont("Arial", 35)
        self.bar_up    = Bar(self.display, (400, 40, 40, 120), self.font)
        self.bar_down  = Bar(self.display, (450, 40, 40, 120), self.font)
        self.bar_left  = Bar(self.display, (500, 40, 40, 120), self.font)
        self.bar_right = Bar(self.display, (550, 40, 40, 120), self.font)
        self.bar_B     = Bar(self.display, (600, 40, 40, 120), self.font)
        self.bar_A     = Bar(self.display, (650, 40, 40, 120), self.font)
 
        ''' Create a list of balloons and set the number '''    
        self.balloons = []
        self.noBalloon = 10

        self.is_neutral = True
        self.detect_time = time.time()
        self.classes = ['Up','Down','Left','Right','B','A']
        ''' Insert balloons into list using for loop '''
        for i in range(self.noBalloon):
            obj = Balloon(random.choice([1, 1, 2, 2, 2, 2, 3, 3, 3, 4]), self.display, self.margin, self.width, self.height, self.lowerBound)
            self.balloons.append(obj)

        
    ''' Control cursor to pop balloon '''
    def pointer(self):
        pos = pygame.mouse.get_pos()
        r = 25
        l = 20
        color = red
        for i in range(self.noBalloon):
            if isonBalloon(self.balloons[i].x, self.balloons[i].y, self.balloons[i].a, self.balloons[i].b, pos):
                color = red
        pygame.draw.ellipse(self.display, color, (pos[0] - r/2, pos[1] - r/2, r, r), 4)
        pygame.draw.line(self.display, color, (pos[0], pos[1] - l/2), (pos[0], pos[1] - l), 4)
        pygame.draw.line(self.display, color, (pos[0] + l/2, pos[1]), (pos[0] + l, pos[1]), 4)
        pygame.draw.line(self.display, color, (pos[0], pos[1] + l/2), (pos[0], pos[1] + l), 4)
        pygame.draw.line(self.display, color, (pos[0] - l/2, pos[1]), (pos[0] - l, pos[1]), 4)

    ''' Create subplatform '''
    def lowerPlatform(self):
        pygame.draw.rect(self.display, darkGray, (0, self.height - self.lowerBound, self.width, self.lowerBound))
        
    ''' Show score on screen '''
    def showScore(self):
        scoreText = self.font.render("Balloons Bursted : " + str(self.score), True, white)
        self.display.blit(scoreText, (150, self.height - self.lowerBound + 50))
        
    ''' Create function to close the game '''
    def close(self):
        pygame.quit()
        sys.exit()
    

    def setPredicts(self, predicts):
        self.predict_up, self.predict_down, self.predict_left, self.predict_right, self.predict_B, self.predict_A = predicts
        max_value = max([self.predict_up, self.predict_down, self.predict_left, self.predict_right, self.predict_B, self.predict_A])

        # ニュートラル
        if max_value <= 0.5:
            if self.is_neutral != True:
                # 今、非ニュートラルならニュートラル描画
                pygame.event.post(self.events[0])
            self.is_neutral = True
            return

        if self.is_neutral:
            if time.time() - self.detect_time < float(5 / 10):
                self.is_neutral = False
                return

            max_index = [self.predict_up, self.predict_down, self.predict_left, self.predict_right, self.predict_B, self.predict_A].index(max_value)
            print(max_value,self.classes[max_index])
            pygame.event.post(self.events[max_index + 1])
            self.detect_time = time.time()

    def run(self):
        #global score
        loop = True
        is_visualization = False
        while loop:
            for event in pygame.event.get():
                ''' End the game only when the 'quit' button is pressed '''
                if event.type == pygame.QUIT:
                    close()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.close()
                    if event.key == pygame.K_r:
                        self.score = 0
                        self.run()

                    if event.key == pygame.K_w:
                        pygame.event.post(self.E_UP)

                    if event.key == pygame.K_s:
                        pygame.event.post(self.E_DOWN)

                    if event.key == pygame.K_a:
                        pygame.event.post(self.E_LEFT)

                    if event.key == pygame.K_d:
                        pygame.event.post(self.E_RIGHT)

                    if event.key == pygame.K_z:
                        pygame.event.post(self.E_B)

                    if event.key == pygame.K_c:
                        pygame.event.post(self.E_A)

                    if event.key == pygame.K_SPACE:
                        pygame.event.post(self.E_NEUTRAL)

                    if event.key == pygame.K_v:
                        is_visualization = ~is_visualization



                if event == self.E_NEUTRAL:
                        self.bg = self.bg_neutral

                if event == self.E_UP:
                        self.bg = self.bg_up

                if event == self.E_DOWN:
                        self.bg = self.bg_down

                if event == self.E_LEFT:
                        self.bg = self.bg_left

                if event == self.E_RIGHT:
                        self.bg = self.bg_right

                if event == self.E_B:
                        self.bg = self.bg_B

                if event == self.E_A:
                        self.bg = self.bg_A



                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i in range(self.noBalloon):
                        self.score = self.balloons[i].burst(self.score)

            self.display.fill(white)
            
            for i in range(self.noBalloon):
                self.balloons[i].show()

            self.pointer()
            
            for i in range(self.noBalloon):
                self.balloons[i].move()

            self.display.blit(self.bg, (0, 0))

            if is_visualization:
                self.bar_up.value = self.predict_up * 100
                self.bar_up.draw(self.display,'U')
                self.bar_down.value = self.predict_down * 100
                self.bar_down.draw(self.display,'D')
                self.bar_left.value = self.predict_left * 100
                self.bar_left.draw(self.display,'L')
                self.bar_right.value = self.predict_right * 100
                self.bar_right.draw(self.display,'R')
                self.bar_B.value = self.predict_B * 100
                self.bar_B.draw(self.display,'B')
                self.bar_A.value = self.predict_A * 100
                self.bar_A.draw(self.display,'A')


            #self.lowerPlatform()
            #self.showScore()
            pygame.display.update()
            self.clock.tick(60)


if __name__=="__main__":

    E_NEUTRAL = pygame.event.Event(pygame.USEREVENT, attr1='E_NEUTRAL')
    E_UP = pygame.event.Event(pygame.USEREVENT, attr1='E_UP')
    E_DOWN = pygame.event.Event(pygame.USEREVENT, attr1='E_DOWN')
    E_LEFT = pygame.event.Event(pygame.USEREVENT, attr1='E_LEFT')
    E_RIGHT = pygame.event.Event(pygame.USEREVENT, attr1='E_RIGHT')
    E_B = pygame.event.Event(pygame.USEREVENT, attr1='E_B')
    E_A = pygame.event.Event(pygame.USEREVENT, attr1='E_A')
    game = Game([E_NEUTRAL, E_UP, E_DOWN, E_LEFT, E_RIGHT, E_B, E_A])
    game.run()
