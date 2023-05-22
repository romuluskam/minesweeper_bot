import os
import pygame
from time import sleep
import numpy as np
from tensorflow.keras.models import load_model
from game import Game
from agent import Agent


# Класс guiGame - главный класс визуальной игры.
# Занимается отрисовкой всех спрайтов, обработкой нажатий мыши и все-все-все.
class guiGame:
    def __init__(self, game, agent=None):
        # Это по сути объект консольной версии игры, которая хранит поле и прочее.
        self.board = game
        self.agent = agent          # Наш агент, который будет предсказывать лучший ход
        pygame.init()
        # параметры для отрисовки главного окна игры
        self.pieceSize = 50
        self.sizeScreen = (self.board.board.cols*self.pieceSize,
                           self.board.board.rows*self.pieceSize)
        self.screen = pygame.display.set_mode(self.sizeScreen)
        pygame.display.set_caption('Minesweeper')
        # прочитаем и загрузим в память все спрайты ячеек
        self.loadResources()
        # т.к. в консольной версии флаг, что здесь мина, не требовался - пришлось его реализовать здесь
        # этот массив хранит расположение флажков на поле
        self.flags = np.zeros(
            (self.board.board.rows, self.board.board.cols)).astype('bool')
        self.ai_choice = None

    # функция для считывания и загрузки в память спрайтов для ячеек поля
    def loadResources(self):
        self.images = {}
        if os.environ.get('enviroment') == 'colab':
            self.imagesDirectory = "/content/drive/MyDrive/NN/Diplom/Minesweeper/images"
            self.soundsDirectory = "/content/drive/MyDrive/NN/Diplom/Minesweeper/sounds"
        else:
            self.imagesDirectory = "./images"
            self.soundsDirectory = "./sounds"
        for fileName in os.listdir(self.imagesDirectory):
            if not fileName.endswith(".png"):
                continue
            path = self.imagesDirectory + r"/" + fileName
            img = pygame.image.load(path)
            img = img.convert()
            img = pygame.transform.scale(
                img, (int(self.pieceSize), int(self.pieceSize)))
            self.images[str(fileName.split(".")[0])] = img

    # главный цикл игры
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                # обработка закрытия окна приложения
                if event.type == pygame.QUIT:
                    running = False
                # обработка нажатия кнопок мыши
                if event.type == pygame.MOUSEBUTTONDOWN and not (self.board.board.is_game_finished or self.board.board.is_game_over):
                    rightClick = pygame.mouse.get_pressed(num_buttons=3)[2]
                    self.handleClick(pygame.mouse.get_pos(), rightClick)
                # обработка нажатия на "Пробел" - для получения предсказания нашего агента
                if event.type == pygame.KEYDOWN:
                    key = pygame.key.get_pressed()
                    if key[pygame.K_SPACE]:
                        self.ai_move()
            # отрисовка графики
            self.screen.fill((0, 0, 0))
            self.draw()
            pygame.display.flip()
            if self.board.board.is_game_finished:
                pygame.display.set_caption('Minesweeper: Victory!')
                self.win()
                running = False
            if self.board.board.is_game_over:
                pygame.display.set_caption('Minesweeper: Game Over!')
                self.lose()
                running = False
        pygame.quit()

    # функция отрисовки графики
    def draw(self):
        topLeft = (0, 0)
        for idx, row in enumerate(self.board.board.array):
            for idy, cell in enumerate(row):
                rect = pygame.Rect(topLeft, (self.pieceSize, self.pieceSize))
                if self.flags[idx, idy]:
                    image = self.images['f']
                else:
                    image = self.images[str(cell)]
                self.screen.blit(image, topLeft)
                topLeft = topLeft[0] + self.pieceSize, topLeft[1]
            topLeft = (0, topLeft[1] + self.pieceSize)
        if self.ai_choice:
            image = self.images['ai']
            topLeft = (self.ai_choice[1]*self.pieceSize,
                       self.ai_choice[0]*self.pieceSize)
            self.screen.blit(image, topLeft)

    # обработчки нажатия кнопки мыши
    def handleClick(self, position, rightClickflag):
        self.ai_choice = None
        # вычисление выбранной клетки
        index = tuple(int(pos // size) for pos, size in zip(position,
                      (self.pieceSize, self.pieceSize)))[::-1]
        if rightClickflag and self.board.board.array[index[0], index[1]] == -1:
            # выставляем/снимаем флажок на щелчок правой кнопкой мыши
            self.flags[index[0], index[1]] = ~self.flags[index[0], index[1]]
        elif not rightClickflag:
            # открываем ячейку на щелчок левой кнопкой мыши
            self.board.open_tile(index[0], index[1])

    # воспроизведение победной мелодии
    def win(self):
        sound = pygame.mixer.Sound(
            self.soundsDirectory + '/win.wav')
        sound.play()
        sleep(3)

    # воспроизведение взрыва - проигрыш
    def lose(self):
        sound = pygame.mixer.Sound(
            self.soundsDirectory + '/lose.mp3')
        sound.play()
        sleep(3)

    # фукнция вызова лучшего хода от агента
    def ai_move(self):
        if self.agent:
            action = self.agent.get_action(best_move=True)
            self.ai_choice = [action // self.board.board.cols,
                              action % self.board.board.cols]
        else:
            print('No agent for AI move!')


if __name__ == '__main__':
    game = Game(rows=9, cols=9, mines=10)
    g = guiGame(game, Agent(game))
    g.run()
