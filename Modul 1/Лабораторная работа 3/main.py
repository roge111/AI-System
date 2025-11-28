from pyswip import Prolog
import FindParent
import FindGrandparents
import Cousins
import Unmarriage
import UnmarriageParent

prolog = Prolog()
prolog.consult('Simple.pl')

patterns = ['Кто родитель', 'Являються ли имя1 и имя2 братьями или сестрами?','Кто прародители', 'Разведен', 'Разведена', 'Разведены ли родители ребенка имя1' ]
print('Введите запрос. Используйте шаблоны!!!')
print(' '.join(patterns))
while True:
    req = input('Введите запрос(Для выхода нажмите q, а затем enter)>> ')
    if req == 'q':
        break
    if patterns[0] in req:
        print(f"Родителями являються: {FindParent.find(prolog)}")
    elif 'Являються ли' in req and 'братьями или сестрми':
        result = Cousins.find()
        if result:
            print('Да')
        else:
            print('Нет')
    elif patterns[2] in req:
        print(f"Прародителями являються: {FindGrandparents.find()}")
    elif patterns[3] in req or patterns[4] in req:
        name1, name2  = input('Введите имена через пробел>> ').split()
        if Unmarriage.find(name1, name2):
            print("Да")
        else:
            print("нет")
    elif patterns[5].replace('имя1', '') in req:
        print(UnmarriageParent.find())
    else:
        print('Такого запроса нет')

