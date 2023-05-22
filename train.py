import os
import pickle
from tqdm import tqdm
import random
import math
import numpy as np
from game import Game
from agent import Agent


if not os.environ.get('enviroment'):
    os.environ['enviroment'] = 'local'
AGG_STATS_EVERY = 100  # вывод статистики каждые 100 эпизодов
SAVE_MODEL_EVERY = 300  # сохранение весов в файл каждые 300 эпизодов
MEM_SIZE = 50_000  # размер буфера памяти
MEM_SIZE_MIN = 1_000  # минимальный размер для запуска обучения
rows = 9
cols = 9
mines = 10
episode = 1
episodes = 100_000  # обучение на 100.000 игровых эпизодов
wins_list, ep_rewards = [], []

game = Game(rows, cols, mines)
agent = Agent(game)

# восстановление обучения
if os.environ['enviroment'] == 'colab':
    PATH_TO_MODEL = f'/content/drive/MyDrive/NN/Diplom/Minesweeper/Release/{agent.model_name}.h5'
    PATH_TO_STATS = f'/content/drive/MyDrive/NN/Diplom/Minesweeper/Release/{agent.model_name}.bin'
else:
    # local
    PATH_TO_MODEL = f'./models/{agent.model_name}.h5'
    PATH_TO_STATS = f'./models/{agent.model_name}.bin'
# загрузка весов
if (os.path.exists(PATH_TO_MODEL)):
    agent.model.load_weights(PATH_TO_MODEL)
# загрузка результатов обучения
if (os.path.exists(PATH_TO_STATS)):
    with open(PATH_TO_STATS, 'rb') as f:
        agent.replay_memory, episode, wins_list, ep_rewards, agent.epsilon = pickle.load(
            f)

for episode in tqdm(range(episode, episodes+1), unit=' episode'):
    mines = random.randint(math.floor(0.07*rows*cols),
                           math.floor(0.23*rows*cols))
    game.reset(mines=mines, first_tile_opened=True)
    ep_reward = 0
    game_ended = False
    while not game_ended:
        # определяем координаты ячейки, которую будем открывать
        current_state = game.board_array
        action = agent.get_action()
        x, y = action//cols, action % cols

        # подаем действие в среду и получаем отклик
        new_state, reward, game_ended = game.open_tile(x, y)
        ep_reward += reward

        # сохраняем в буфер памяти
        agent.update_replay_memory(
            (current_state, action, reward, new_state, game_ended))
        agent.train(game_ended)

    ep_rewards.append(ep_reward)
    wins_list.append(1 if game.board.is_game_finished else 0)

    # выводим статистику
    if not episode % AGG_STATS_EVERY:
        win_rate = np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY * 100
        med_reward = np.median(ep_rewards[-AGG_STATS_EVERY:])
        print(
            f'Episode: {episode}, Avg reward: {med_reward:.1f}, Winrate: {win_rate:.2f}%, Epsilon: {agent.epsilon:.5f}')

    # сохраняем веса модели и буфер памяти в файл
    if not episode % SAVE_MODEL_EVERY:
        with open(PATH_TO_STATS, 'wb') as output:
            pickle.dump([agent.replay_memory, episode, wins_list,
                        ep_rewards, agent.epsilon], output)
        with open(PATH_TO_STATS+'_', 'wb') as output:
            pickle.dump([agent.replay_memory, episode, wins_list,
                        ep_rewards, agent.epsilon], output)
        agent.model.save_weights(PATH_TO_MODEL)
        agent.model.save_weights(PATH_TO_MODEL+'_')
