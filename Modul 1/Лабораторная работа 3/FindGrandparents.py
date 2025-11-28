from pyswip import Prolog
def find():
    prolog = Prolog()
    prolog.consult('Simple.pl')

    name = input("Введите именя еще раз?")

    result = []
    for res in prolog.query(f"is_grandparent(X, {name})"):
        if res['X'] not in result:
            result.append(res["X"])
    return result

