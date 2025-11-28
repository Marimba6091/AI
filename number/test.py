from normalize_data import Matrix_to_img
dct = {4:"0,0,0,0,1,0,0,0,0,0", 2:"0,0,1,0,0,0,0,0,0,0", 0:"1,0,0,0,0,0,0,0,0,0", 3:"0,0,0,1,0,0,0,0,0,0", 1:"0,1,0,0,0,0,0,0,0,0", 5:"0,0,0,0,0,1,0,0,0,0", 6:"0,0,0,0,0,0,1,0,0,0", 7:"0,0,0,0,0,0,0,1,0,0", 8:"0,0,0,0,0,0,0,0,1,0", 9:"0,0,0,0,0,0,0,0,0,1"}

def rer(x):
    for i in range(10):
        if x == dct[i]:
            return i

with open("number/data.txt", "r") as file:
    for x, line in enumerate(file):
        num = line.replace("\n", "").split(" ")[1]
        Matrix_to_img(list(map(int, line.split(" ")[0].split(","))), out=f"img/out_{rer(num)}_{x}.png")