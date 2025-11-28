from pyswip import Prolog
import warnings

def find(prolog):
    prolog = Prolog()
    prolog.consult('Simple.pl')
    name = input("Введите имя>> ")

    result = []
    for res in prolog.query(f'parent(X, {name})'):
        
        result.append(res['X'])
    return result