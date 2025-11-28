from tkinter import *
from tkinter.messagebox import *
import io
import numpy as np
from PIL import ImageGrab
from normalize_data import Img_to_matrix, Matrix_to_img, out_for_data
from Neuron_net import NN

def painting(event):
    global display
    x = event.x
    y = event.y
    display.create_oval(x-5, y-5, x+5, y+5, fill="#000000")

def clear(event=None):
    global display
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
    l.config(text=f"{np.argmax(predict)}")

def save_data(event):
    if askyesno(title="Подтвержение операции", message="Подтвердить операцию?"):
        x = win.winfo_rootx() + display.winfo_x()
        y = win.winfo_rooty() + display.winfo_y()
        x1 = x + display.winfo_width()
        y1 = y + display.winfo_height()
        with open("number/data.txt", "a") as file:
            file.write(f"\n{",".join(list(map(str, Img_to_matrix(ImageGrab.grab().crop((x+5, y+5, x1-5, y1-5))).get_data())))} {out_for_data[int(event.keysym)]}")

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

win = Tk()
win.title("Canvas")
win.geometry("350x550")
win.resizable(0, 0)
win.config(bg="#C6BC96")

nn = NN()

display = Canvas(width=350, height=350, bg="#FFFFFF")
display.pack()

l = Label(text="none", font=("Arial", 25), bg="#C6BC96")
l.pack()

win.bind("<B1-Motion>", painting)
win.bind('<BackSpace>', clear)
win.bind("<Return>", get_img)
win.bind("<Key-1>", save_data)
win.bind("<Key-2>", save_data)
win.bind("<Key-3>", save_data)
win.bind("<Key-4>", save_data)
win.bind("<Key-5>", save_data)
win.bind("<Key-6>", save_data)
win.bind("<Key-7>", save_data)
win.bind("<Key-8>", save_data)
win.bind("<Key-9>", save_data)
win.bind("<Key-0>", save_data)
win.bind("<space>", train_nn)

win.mainloop()