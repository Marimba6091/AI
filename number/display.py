from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk
import numpy as np
from PIL import ImageGrab
from normalize_data import Img_to_matrix, out_for_data
from Neuron_net import NN
import keyboard as kb

def painting(event):
    global display
    x = event.x
    y = event.y
    display.create_oval(x-5, y-5, x+5, y+5, fill="#000000")

def clear(event=None):
    global display
    l.config(text="none")
    for i in progress:
        i[1]["value"] = 0
    display.delete("all")

def get_img(event=None):
    if not nn.is_set_data:
        nn.set_data_from("number/train_data.json")
    x = win.winfo_rootx() + display.winfo_x()
    y = win.winfo_rooty() + display.winfo_y()
    x1 = x + display.winfo_width()
    y1 = y + display.winfo_height()
    img = ImageGrab.grab().crop((x+5, y+5, x1-5, y1-5))
    img.save("input.png")
    matrix = np.array(Img_to_matrix(img).get_data())
    predict = nn.predict(matrix)
    if sum(predict) > .4:
        l.config(text=f"{np.argmax(predict)}")
    else:
        l.config(text="none")
    for x, y in enumerate(predict):
        progress[x][1]["value"] = int(y * 100)

def save_data(event):
    if askyesno(title="Подтвержение операции", message=f"Это действительно {event.keysym}?"):
        x = win.winfo_rootx() + display.winfo_x()
        y = win.winfo_rooty() + display.winfo_y()
        x1 = x + display.winfo_width()
        y1 = y + display.winfo_height()
        matrix = Img_to_matrix(ImageGrab.grab().crop((x+5, y+5, x1-5, y1-5))).get_data()
        if matrix:
            with open("number/data.txt", "a") as file:
                file.write(f"\n{",".join(list(map(str, matrix)))} {out_for_data[int(event.keysym)]}")

def train_nn(event):
    ask = askyesnocancel(title="По вопросам образования", message="Продолжить обучение с нынешними данными?")
    if not ask is None:
        if ask:
            nn.set_data_from("number/train_data.json")
        file = open("number/data.txt")
        file_data = []
        for i in file:
            f = i.split(" ")
            try:
                file_data.append((np.array(list(map(int, f[0].split(",")))), (np.array(list(map(int, f[1].split(",")))))))
            except:
                print(f[0], f[1])
        nn.train(file_data, epochs=1000, lmd=0.1, clear=ask)

def enter_canvas(event):
    global on_canvas
    on_canvas = True

def leave_canvas(event):
    global on_canvas
    on_canvas = False

win = Tk()
win.title("Canvas")
win.geometry("350x650")
win.resizable(0, 0)
win.config(bg="#C6BC96")

nn = NN()

on_canvas = False

display = Canvas(width=350, height=350, bg="#FFFFFF")
display.pack()

l = Label(text="none", font=("Arial", 25), bg="#C6BC96")
l.pack()

progress = []
F_progress = Frame(bg="#C6BC96")
F_progress.pack()
for i in range(10):
    progress.append([Label(F_progress, text=f"{i}", bg="#C6BC96"), ttk.Progressbar(F_progress, length=150)])
    progress[-1][0].grid(column=0, row=i)
    progress[-1][1].grid(column=1, row=i)

win.bind("<B1-Motion>", painting)
win.bind('<BackSpace>', clear)
win.bind("<Return>", get_img)
win.bind("<space>", train_nn)
display.bind("<Enter>", enter_canvas)
display.bind("<Leave>", leave_canvas)
for i in range(10):
    win.bind(f"<Key-{i}>", save_data)

win.mainloop()