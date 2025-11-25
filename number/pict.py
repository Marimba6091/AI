import os
import normalize_data as nrd
from PIL import Image

file = "input.png"
dct = {4:"0,0,0,0,1,0,0,0,0,0", 2:"0,0,1,0,0,0,0,0,0,0", 0:"1,0,0,0,0,0,0,0,0,0", 3:"0,0,0,1,0,0,0,0,0,0"}
number = 3
b = os.path.getsize(file)
while True:
    if b != os.path.getsize(file):
        try:
            img = Image.open(file)
            b = os.path.getsize(file)
            with open("number/data.txt", "a") as data:
                data.write(f"\n{",".join(list(map(str, nrd.Img_to_matrix(img).get_data())))} {dct[number]}")
            img.close()
            print(number)
        except:
            print("error")
