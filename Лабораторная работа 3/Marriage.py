from pyswip import Prolog

def find():
    prolog = Prolog
    prolog.consult('Simple.pl')

    name1, name2  = input('Введите имена через пробел>> ').split()
    result = []
    for res in prolog.query(f'marriage({name1}, {name2}).'):
        result.append(res)
        break
    return len(result) > 0
