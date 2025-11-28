file1 = open('Программа.txt', encoding='utf-8')
file2 = open('Simple.pl', 'w', encoding='utf-8')

for s in file1:
    s = s.lower()
    file2.write(s)
    