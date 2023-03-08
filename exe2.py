C = "asvphgyt"

def Kfinder():
    def increaseChar(k: int):
        return list(map(lambda x: chr((ord(x) - 96 + k) % 26 + 96), [*C]))

    return list(map(lambda x: "".join(increaseChar(x)), range(0, 26)))

for i in range(0, len(Kfinder())):
    print(f' k = {i}, plaintext: {Kfinder()[i]}')


