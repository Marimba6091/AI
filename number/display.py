from tkinter import *
import io
import numpy as np
from PIL import ImageGrab
from normalize_data import Img_to_matrix, Matrix_to_img
from Neuron_net import NN

def painting(event):
    global display
    x = event.x
    y = event.y
    display.create_oval(x-10, y-10, x+10, y+10, fill="#000000")

def clear(event=None):
    global display
    display.delete("all")

def get_img(event=None):
    x = win.winfo_rootx() + display.winfo_x()
    y = win.winfo_rooty() + display.winfo_y()
    x1 = x + display.winfo_width()
    y1 = y + display.winfo_height()
    img = ImageGrab.grab().crop((x+5, y+5, x1-5, y1-10))
    img.save("input.png")
    matrix = np.array(Img_to_matrix(img).get_data())
    Matrix_to_img(matrix)
    nn = NN()
    nn.set_data_from("number/train_data.json")
    predict = nn.predict(matrix)
    l.config(text=f"{np.argmax(predict)}")


win = Tk()
win.title("Canvas")
win.geometry("350x550")
win.resizable(0, 0)

display = Canvas(width=350, height=350, bg="#FFFFFF")
display.pack()

l = Label(text="none", font=("Arial", 25))
l.pack()

win.bind("<B1-Motion>", painting)
win.bind('<BackSpace>', clear)
win.bind("<Return>", get_img)

win.mainloop()