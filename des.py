from functools import reduce
import numpy as np

"""
Step 0
Preprocessing
"""
# Input for hex format
hexM = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF]
hexK = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0x5, 0x9, 0x7, 0x2]

# Convert hex format into binary format
binM = list(map(lambda x: bin(x), hexM))
binK = list(map(lambda x: bin(x), hexK))

# Convert binary number format into string format
binM = list(map(lambda x: [int(char) for char in x[2:]] if len(x[2:]) == 4 else [0 * i for i in range(4 - len(x[2:]))] +
                                                                                [int(char) for char in x[2:]], binM))

binK = list(map(lambda x: [int(char) for char in x[2:]] if len(x[2:]) == 4 else [0 * i for i in range(4 - len(x[2:]))] +
                                                                                [int(char) for char in x[2:]], binK))

# Flatten list of number as list
M = list(reduce(lambda prev, curr: prev + curr, binM, []))
K = list(reduce(lambda prev, curr: prev + curr, binK, []))

"""
Step 1
Create 16 subkeys
"""
PC1 = [57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18, 10, 2, 59, 51, 43, 35, 27,
       19, 11, 3, 60, 52, 44, 36, 63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22,
       14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4]

PC2 = [14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10, 23, 19, 12, 4, 26, 8,
       16, 7, 27, 20, 13, 2, 41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48,
       44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32]

# Calculate Kplus
Kplus = [K[i - 1] for i in PC1]

# Calculate (C0, D0) to (Cn, Dn)
numOfLeftShift = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

# Initialize first element (C0, D0)
C, D = Kplus[:28], Kplus[28:]
CD = []

# Generate (C0, D0) to (Cn, Dn)
for i in numOfLeftShift:
    C, D = list(np.roll(C, -i)), list(np.roll(D, -i))
    CD.append((C + D))

# Generate K from CD
K = [[]] + [[CDi[j - 1] for j in PC2] for CDi in CD]
print(f'K1: {K[1]}')
"""
Step 2
Encode
"""

IP = [58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4, 62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48,
      40, 32, 24, 16, 8, 57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3, 61, 53, 45, 37, 29,
      21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7]

s_boxes = [[
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
    [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
    [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
    [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
], [
    [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
    [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
    [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
    [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
], [
    [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
    [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
    [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
    [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
], [
    [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
    [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
    [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
    [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
], [
    [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
    [4, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
    [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
    [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
], [
    [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
    [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
    [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
    [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
], [
    [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
    [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
    [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
    [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
], [
    [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
    [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
    [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
    [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
]]

P = [16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25]

IP_INV = [40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
          38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
          36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
          34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25]

IPM = [M[IPi - 1] for IPi in IP]
L0, RO = IPM[:32], IPM[32:]
print(f'LO: {L0}')
print(f'R0: {RO}')


def E(R):
    result = []
    for i in range(0, 32, 4):
        if i == 0:
            result += [R[31]] + R[:4] + [R[4]]
        elif i == 28:
            result += [R[27]] + R[i:] + [R[0]]
        else:
            result += [R[i - 1]] + R[i:i + 4] + [R[i + 4]]
    return result


print(f'E(R0): {E(RO)}')


def xorfunc(e, k):
    return list(map(lambda x, y: x ^ y, e, k))


print(f'A = xor(E(R0), K1) = {xorfunc(E(RO), K[1])}')


def sBoxfunc(x):
    result = []
    for i in range(0, 8):
        childx = x[6 * i:6 * (i + 1)]
        matched = s_boxes[i][childx[0] * 2 + childx[5]][childx[1] * 8 + childx[2] * 4 + childx[3] * 2 + childx[4]]
        binaryconverted = bin(matched)[2:].zfill(4)
        result += [int(bit) for bit in binaryconverted]
    return result


print(f'B = {sBoxfunc(xorfunc(K[1], E(RO)))}')


def permuatationfunc(x):
    return [x[i - 1] for i in P]


print(f'P(B) = {permuatationfunc(sBoxfunc(xorfunc(K[1], E(RO))))}')


def caculateRfunc(prevL, prevR, i):
    R1 = E(prevR)
    R2 = xorfunc(K[i], R1)
    R3 = sBoxfunc(R2)
    R4 = permuatationfunc(R3)
    R5 = list(map(lambda x, y: x ^ y, prevL, R4))
    return R5

R1 = caculateRfunc(L0, RO, 1)
print(f'R1 = {R1}')


def convertarraybintoValue(a):
    result = ""
    for i in range(0, 64, 4):
        bits = a[i:i + 4]
        binary_string = ''.join(map(str, bits))  # Convert binary values to string
        decimal_value = int(binary_string, 2)  # Convert binary string to decimal integer
        hex_string = str(hex(decimal_value)[2:].upper())
        result += hex_string
    return result

round1 = RO + R1
print(f'Ciphertext round1: {convertarraybintoValue(round1)}')


L, R = L0, RO
for i in range(1, 17):
    L, R = R, caculateRfunc(L, R, i)
R16L16 = R + L
ip_inv = [R16L16[i - 1] for i in IP_INV]
finalresult = convertarraybintoValue(ip_inv)
print(f'Final result: {finalresult}')
