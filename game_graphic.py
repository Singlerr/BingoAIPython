import asyncio
import tkinter as tk

import train
import game


class GameGraphicManager(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.board = [[None] * 5] * 5
        self.position = [(None, None) for _ in range(25)]
        self.grid()
        self.create_board()
        self.updater()
    def create_board(self):
        k = 0
        for i, row in enumerate(self.board):
            for j, col in enumerate(row):
                self.position[k] = (i, j)
                k += 1
                label = tk.Label(self, text='', width=5, height=2, pady=3, padx=3, fg="red", relief="solid")
                label.grid(row=i, column=j)
                self.board[i][j] = label

    def update_board(self):
        for i in range(25):
            val = game.gameState.squares[i]
            (x, y) = self.position[i]
            self.board[x][y]["text"] = str(val)

    def updater(self):
        self.update_board()
        self.after(100, self.updater)
