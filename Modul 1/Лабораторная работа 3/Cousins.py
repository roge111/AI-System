from pyswip import Prolog

def find():
    name1, name2 = input("Введите имена через пробел еще раз>> ").split()
    prolog = Prolog()
    prolog.consult('Simple.pl')

    result = []
    for res in prolog.query(f'is_cousins({name1}, {name2}).'):
        result.append(res)
        break
    return len(result) > 0

