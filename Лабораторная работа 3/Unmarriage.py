from pyswip import Prolog

def find(name1, name2):
    prolog = Prolog
    prolog.consult('Simple.pl')

    
    result = []
    for res in prolog.query(f'unmarriage({name1}, {name2}).'):
        result.append(res)
        break
    return len(result) > 0
