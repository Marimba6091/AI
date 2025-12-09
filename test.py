import copy


def is_digit(x):
    y = copy.copy(x)
    result = None
    if y[0] in ["-", '+']:
        y = y[1:]
    if y.count("."):
        y = y.split(".")
        y1 = y[0]
        y2 = y[1]
        result = True if y[0].isdigit() and y[1].isdigit() else False
    else:
        result = y.isdigit()
    return result

def numerate(x):
    y = copy.copy(x)
    if "." in y:
        dct = {}
        y = y.split(".")
        dct |= enumerate(reversed(y[0]))
        for i, k in enumerate(y[1]):
            dct[-(i+1)] = k
        return dct.items()
    return enumerate(reversed(x))

def int_(input_, y, dct={}):
    x = input_.lower()
    out = 0
    if y == 16:
        dct |= {"a":10, "b":11, "c":12, "d":13, "e":14, "f":15}
    for i, l in numerate(x):
        if is_digit(l):
            if 0 <= int(l) < y:
                out += int(l) * y ** i
            else:
                raise ValueError(f"{x} : {l} - It's no in S.S. : Position - {i}")
        else:
            if dct:
                out += dct[l] * y ** i
            else:
                raise ValueError(f"{x} : {l} - It's no in dictionary! : Position - {i}")
    return out
print(int_("aa", 16))