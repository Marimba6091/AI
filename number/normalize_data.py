from PIL import Image

class Img_to_matrix:
    def __init__(self, img):
        self.__img = img
        self.__width, self.__heigth = self.__img.size
    
    def get_data(self):
        self.__img = self._get_last_pxl()
        self.__img = self.__img.resize((16, 16))
        self.__img = self._convert()
        return self._cross()

    def _convert(self):
        if not isinstance(self.__img.getpixel((0, 0)), int):
            for w in range(16):
                for h in range(16):
                    pxl = self.__img.getpixel((w, h))
                    if abs(pxl[0] - pxl[1] - pxl[2]) == pxl[0] and pxl[0] < 210:
                        self.__img.putpixel((w, h), (0, 0, 0))
                    else:
                        self.__img.putpixel((w, h), (255, 255, 255))
        return self.__img.convert("1")

    def _get_last_pxl(self):
        fw, fh, lw, lh = self.__width,self.__heigth,0,0

        for w in range(self.__width):
            for h in range(self.__heigth):
                pxl = self.__img.getpixel((w, h))
                if pxl[0] + pxl[1] + pxl[2] < 765:
                    fw = w if w < fw else fw
                    fh = h if h < fh else fh

        for w in reversed(range(self.__width)):
            for h in reversed(range(self.__heigth)):
                pxl = self.__img.getpixel((w, h))
                if pxl[0] + pxl[1] + pxl[2] < 765:
                    lw = w if w > lw else lw
                    lh = h if h > lh else lh

        return self.__img.crop((fw, fh, lw, lh))

    def _not_bool(self, x):
        return not bool(x)

    def _cross(self):
        width, heigth = 16, 16
        matrix = []
        d_m = []
        for h in range(0, heigth, 2):
            for w in range(0, width, 2):
                d_m = tuple(map(self._not_bool, [self.__img.getpixel((w, h)), self.__img.getpixel((w+1, h)), self.__img.getpixel((w, h+1)), self.__img.getpixel((w+1, h+1))]))
                matrix.append(d_m)
        for number, group in enumerate(matrix):
            matrix[number] = int("".join(tuple(map(str, map(int, group)))), 2)
        return matrix

class Matrix_to_img:
    def __init__(self, matrix, out="out.png"):
        self.__matrix = matrix
        self.__out = out
        self.__create()
    
    def _bool_to_int(self, x):
        return 255 if not x else 0
    
    def _to_binary(self, x):
        return list(map(int, bin(x)[2:]))
    
    def _correct(self, d):
        for x, group in enumerate(d):
            for i in range(4 - len(group)):
                d[x].insert(0, 255)
        return d
    
    def __create(self):
        img = Image.new(size=(16, 16), mode="1")
        pxls = []
        for i in self.__matrix:
            pxls.append(list(map(self._bool_to_int, self._to_binary(i))))
        pxls = self._correct(pxls)
        for idx, group in enumerate(pxls):
            block_row = idx // 8
            block_col = idx % 8
            img.putpixel((block_col * 2, block_row * 2), group[0])
            img.putpixel((block_col * 2 + 1, block_row * 2), group[1])
            img.putpixel((block_col * 2, block_row * 2 + 1), group[2])
            img.putpixel((block_col * 2 + 1, block_row * 2 + 1), group[3])
        img.save(self.__out)
        
out_for_data = {4:"0,0,0,0,1,0,0,0,0,0", 2:"0,0,1,0,0,0,0,0,0,0", 0:"1,0,0,0,0,0,0,0,0,0", 3:"0,0,0,1,0,0,0,0,0,0", 1:"0,1,0,0,0,0,0,0,0,0", 5:"0,0,0,0,0,1,0,0,0,0", 6:"0,0,0,0,0,0,1,0,0,0", 7:"0,0,0,0,0,0,0,1,0,0", 8:"0,0,0,0,0,0,0,0,1,0", 9:"0,0,0,0,0,0,0,0,0,1"}
if __name__ == "__main__":
    cls = Matrix_to_img([0,0,0,0,0,0,15,13,0,0,0,0,0,0,15,12,0,0,0,0,0,5,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #cls = Img_to_matrix(Image.open("input.png")).get_data()
    #print(f"{",".join(list(map(str, cls)))} {out_for_data[0]}")
