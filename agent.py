from IPython.display import clear_output
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import pickle
from typing import Union, List
import random
import time
import math
import numpy as np
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

# параметры обучения Агента по умолчанию
MEM_SIZE = 50_000  # размер буфера памяти
MEM_SIZE_MIN = 1_000  # минимальный размер для запуска обучения

BATCH_SIZE = 64
LEARNING_RATE = 0.0001
DISCOUNT = 0.1  # gamma

EPSILON = 0.95
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01

# обовляем веса таргет-модели каждые N эпизодов
UPDATE_TARGET_EVERY = 100


# архитектура модели
def NN(learn_rate, input_dims, output_dims):
    model = Sequential([
        Conv2D(256, (3, 3), activation='relu', padding='same',
               input_shape=input_dims),
        Conv2D(256, (3, 3), activation='relu',
               padding='same'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(output_dims, activation='linear')])

    model.compile(optimizer=RMSprop(
        learning_rate=learn_rate, epsilon=1e-4), loss='mse')

    return model


class Agent(object):
    def __init__(self, game):
        self.game = game
        self.model_name = str(self.game.board.rows) + \
            'x' + str(self.game.board.cols)
        self.discount = DISCOUNT
        self.learn_rate = LEARNING_RATE
        self.epsilon = EPSILON
        state_im = self.game.board_array
        state_im3d = state_im[..., np.newaxis]
        # основная (обучаемая) модель
        self.model = NN(
            self.learn_rate, state_im3d.shape, self.game.board.rows*self.game.board.cols)
        # таргет (необучаемая) модель
        self.target_model = NN(
            self.learn_rate, state_im3d.shape, self.game.board.rows*self.game.board.cols)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

    # функция выбора следующего хода
    def get_action(self, best_move=False):
        rand = np.random.random()
        # обманка, чтобы выбрать лучший ход
        if best_move:
            self.epsilon = -1
        if rand < self.epsilon:
            # рандомный ход
            unopened_tiles_idxs = [i for i, x in enumerate(
                self.game.unopened_tiles.flatten()) if x]
            move = np.random.choice(unopened_tiles_idxs)
        else:
            # лучший ход
            moves = self.model.predict(np.expand_dims(
                self.game.board_array, axis=0), verbose=0)
            moves = np.reshape(
                moves, (self.game.board.rows, self.game.board.cols))
            # присваиваем открытым ячейкам минимальную вероятность
            moves[~self.game.unopened_tiles] = np.min(moves)
            move = np.argmax(moves)
        return move

    # добавление в буфер памяти
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # обучение агента
    def train(self, done):
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        batch = random.sample(self.replay_memory, BATCH_SIZE)

        current_states = np.array([transition[0] for transition in batch])
        current_qs_list = self.model.predict(current_states, verbose=0)

        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(
            new_current_states, verbose=0)

        X_train, Y_train = [], []

        for i, (current_state, action, reward, new_current_states, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            X_train.append(current_state)
            Y_train.append(current_qs)

        self.model.fit(np.array(X_train), np.array(Y_train), batch_size=BATCH_SIZE,
                       shuffle=False, verbose=0)

        if done:
            self.target_update_counter += 1

        # обновляем веса таргет-модели
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # уменьшаем эпсилон
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)
