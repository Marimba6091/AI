from normalize_data import Matrix_to_img, data_for_out
import os


def create_img(x, y):
    f = x.split(" ")
    f[0] = list(map(int, f[0].split(",")))
    f[1] = f[1].replace("\n", "")
    Matrix_to_img(f[0], out=f"imgs/img_{data_for_out[f[1]]}_{y}.png")

def main():
    with open("number/data.txt", "r") as file:
        for y, line in enumerate(file):
            create_img(line, y)

def delete():
    for file in os.listdir("imgs"):
        os.remove(f"imgs/{file}")

if not bool(input(">>> ")):
    main()
else:
    delete()