import os
import normalize_data as nrd
from PIL import Image

file = "input.png"
dct = {4:"0,0,0,0,1,0,0,0,0,0", 2:"0,0,1,0,0,0,0,0,0,0", 0:"1,0,0,0,0,0,0,0,0,0", 3:"0,0,0,1,0,0,0,0,0,0", 1:"0,1,0,0,0,0,0,0,0,0", 5:"0,0,0,0,0,1,0,0,0,0", 6:"0,0,0,0,0,0,1,0,0,0", 7:"0,0,0,0,0,0,0,1,0,0", 8:"0,0,0,0,0,0,0,0,1,0", 9:"0,0,0,0,0,0,0,0,0,1"}
number = 4
b = os.path.getsize(file)
while True:
    if b != os.path.getsize(file):
        try:
            img = Image.open(file)
            with open("number/data.txt", "a") as data:
                data.write(f"\n{",".join(list(map(str, nrd.Img_to_matrix(img).get_data())))} {dct[number]}")
            img.close()
            print(number)
            b = os.path.getsize(file)
        except:
            print("error")
