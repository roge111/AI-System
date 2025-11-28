from pyswip import Prolog
import Unmarriage

def find():
    prolog = Prolog
    prolog.consult('Simple.pl')

    name= input('Введите имя> ')
    parents = []

    for parent in prolog.query(f'parent(X, {name})'):
        parents.append(parent['X'])
    name1, name2 = parents[0], parents[1]

    if Unmarriage.find(name1, name2):
        return "Да"
    else:
        return 'Нет'
    