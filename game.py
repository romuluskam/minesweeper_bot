import random
import numpy as np
from board import Board

# Класс игры
# # содержит игровое поле,
# # выдает награды


class Game:
    def __init__(self, rows: int = 10, cols: int = 10, mines: int = 10):
        self._rows = rows
        self._cols = cols
        self._mines = mines
        self.reset()

    def reset(self, mines: int = 0, first_tile_opened=False):
        if not mines:
            mines = self._mines
        self.board = Board(
            rows=self._rows, cols=self._cols, mines=mines)
        if first_tile_opened:
            x = random.randint(0, self._rows - 1)
            y = random.randint(0, self._cols - 1)
            self.board.tile_open(x, y)
            # исключаем случаи, когда игра выигрывается за 1 клик
            if self.board.is_game_finished:
                self.reset(mines=self._mines, first_tile_opened=True)
        return self.board_array

    def open_tile(self, x: int, y: int):
        reward = 0
        game_ended = False
        if not self.board.tile_valid(x, y):
            return self.board_array, -1, game_ended
        has_neighbor = self.board.tile_has_neighbour(x, y)
        tiles = self.board.tile_open(x, y)
        # раздаем награды
        if self.board.is_game_over:
            reward = -1
            game_ended = True
        elif self.board.is_game_finished:
            reward = 1
            game_ended = True
        elif not len(tiles):
            # открыли открытую клетку
            # т.к. частенько бесконечные попытки открыть открытую ячейку приводили
            # к зависанию обучения - пришлось завершать игру при клике на открытую ячейку
            # и давать минимальную награду
            reward = -1
            game_ended = True
        else:  # len(tiles) > 0:
            if has_neighbor:
                reward = 0.3
            else:
                reward = -0.3
        return self.board_array, reward, game_ended

    @property
    def unopened_tiles(self) -> np.ndarray:
        tiles = self.board.array
        return np.where(tiles == -1, True, False)

    @property
    def board_array(self) -> np.ndarray:
        # нормировка (максимальное значение поля = 8)
        return self.board.array/8

    @property
    def board_array_padded(self) -> np.ndarray:
        return np.pad(self.board_array, pad_width=2, mode='constant', constant_values=-1/8)
#         return np.pad(self.board_array, pad_width=2, mode='constant', constant_values=0)

    def split_5x5(self) -> np.ndarray:
        padded = self.board_array_padded
        # попробуем сначала окружить 3, затем еще 0
#         padded = np.pad(self.board_array, pad_width=1, mode='constant', constant_values=-2/8)
#         padded = np.pad(padded, pad_width=1, mode='constant', constant_values=0)
        stack = []
        for i in range(padded.shape[0]-4):
            for j in range(padded.shape[1]-4):
                stack.append(padded[i:i+5, j:j+5])
        return np.array(stack)

    def split_5x5_f(self, mask: np.ndarray) -> np.ndarray:
        # 1. применяем маску
        board = self.board_array
        board[mask] = -2/8
        padded = np.pad(board, pad_width=2, mode='constant',
                        constant_values=-1/8)
        stack = []
        for i in range(padded.shape[0]-4):
            for j in range(padded.shape[1]-4):
                stack.append(padded[i:i+5, j:j+5])
        return np.array(stack)
