from collections import Counter
LETTER_FREQUENCY = {'E': 12.70, 'A': 8.55, 'R': 7.00, 'I': 6.25, 'O': 5.75, 'T': 5.05, 'N': 5.00, 'S': 7.25, 'L': 4.00,
                    'C': 2.75, 'U': 2.75, 'D': 4.25, 'P': 2.50, 'M': 3.00, 'H': 6.25, 'G': 1.75, 'B': 1.50, 'F': 2.25,
                    'Y': 1.75, 'W': 2.25, 'K': 0.75, 'V': 0.75, 'X': 0.15, 'J': 0.20, 'Q': 0.10, 'Z': 0.07}
def letter_frequency(string):
    letters = [c for c in string.upper() if c.isalpha()]    # Convert string to lowercase and remove non-letter characters
    frequency = dict(Counter(letters))      # Count the frequency of each letter
    length = len(string)     # Length of string
    for key, values in frequency.items():     # Convert number of appearance to rate
        frequency[key] = round((values / length) * 100, 2)
    frequency = dict(sorted(frequency.items(), key=lambda item: (item[1], item[0]), reverse=True))     # sort the dictionary by value first, then by key
    return frequency

def find_plaintext(text, num_of_test):
    frequent_letter_table = letter_frequency(text)
    print(f'Letter frequency in English:\t{LETTER_FREQUENCY}')
    print(f'Letter frequency in text:\t{frequent_letter_table}')
    for i in range(num_of_test):
        distance = ord(list(frequent_letter_table.keys())[i]) - ord(list(LETTER_FREQUENCY.keys())[0])
        plaintext = ""
        for c in text:
            new_c = ord(c) - distance
            if new_c < 65:
                new_c = 91 - (65 - new_c)
            elif new_c > 90:
                new_c = 64 + (new_c - 90)
            plaintext = plaintext + chr(new_c)
        print(f'With k = {distance}, the plaintext is "{plaintext}"')

print("Ciphertext: ")
ciphertext = str(input())
print("Number of attempts: ")
numofattempts = int(input())

find_plaintext(ciphertext, numofattempts)
