import tkinter

import game_graphic
import asyncio
import train
import threading

"""
root = tkinter.Tk()
root.title("BINGO")
root.geometry("300x200+100+100")
root.resizable(True, True)
app = game_graphic.GameGraphicManager(root)
root.mainloop()
"""
asyncio.run(train.run_training())
