import os
import random
import numpy as np
from game import Game
from agent import Agent

# начальные параметры игрового поля
rows = 9
cols = 9
mines = 10

# проверка работы игровой среды:
# выставляем рандомные шаги, пока не выиграем/проиграем
rewards_list = []
game_ended = False

game = Game(rows, cols, mines)
game.reset()
while not game_ended:
    x = random.randint(0, cols-1)
    y = random.randint(0, rows-1)
    board, reward, game_ended = game.open_tile(x, y)
    rewards_list.append(reward)
print('Результат игры:', sum(rewards_list), rewards_list)
print(board)


game = Game(rows, cols, mines)
agent = Agent(game)

# # загрузка весов обученной модели
PATH_TO_MODEL = f'/models/{agent.model_name}.h5'
if (os.path.exists(PATH_TO_MODEL)):
    agent.model.load_weights(PATH_TO_MODEL)

rewards = []
game_ended = False

game.reset()
while not game_ended:
    action = agent.get_action(best_move=True)
    x, y = action//rows, action % rows
    board, reward, game_ended = game.open_tile(x, y)
    rewards.append(reward)

game_text_status = 'ПОБЕДА!' if game.board.is_game_finished else 'GAME OVER!'
print(game_text_status)
print('Результат игры:', sum(rewards), rewards)
print(game.board)
