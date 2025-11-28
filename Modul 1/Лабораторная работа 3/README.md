# Лабораторная работа 3 
Выполнил: Батаргин Егор Александрович
Группа: P3332
ИСУ: 335189
# 

Целью этой лабораторной работы является разработка программы (рекомендательной системы), которая будет использовать базу знаний или онтологию для предоставления рекомендаций на основе введенных пользователем данных. (Knowledge-based support system)

**Задание**

- Создать программу, которая позволяет пользователю ввести запрос через командную строку. Например, информацию о себе, своих интересах и предпочтениях в контексте выбора видеоигры - на основе фактов из БЗ (из первой лабы)/Онтологии(из второй).
- Использовать введенные пользователем данные, чтобы выполнить логические запросы к  БЗ/Онтологии.
- На основе полученных результатов выполнения запросов, система должна предоставить рекомендации или советы, связанные с выбором из БЗ или онтологии.
- Система должна выдавать рекомендации после небольшого диалога с пользователем.

# Выполнение

**Main**

main.py - главный файл от которого идет запуск. Работа выполняется на основе базы знаний Prolog. 

Примеры базы заныний:

    %% база людей
    people(egor).
    people(zhanna).
    people(alexandr).
    people(lilya).
    people(mari-sofi).
    people(irina_lee).
    people(vladimir_p).
    people(valdimir_lee).
    people(alexandr_rudometkin).
    people(evelina_rudometkin).

    %% база дат рождения
    birthday(egor, 20-02-2003).
    birthday(zhanna, 21-12-1970).
    birthday(alexandr, 16-08-1964).
    birthday(irina_lee, 13-01-1973).
    birthday(artur_lee, 23-09-1969).
    birthday(vladidmir_p, 17-06-1946).
    birthday(svetlana_p, 29-05-1950).
    birthday(alla_ponomorenko, 14-03-1927).
    birthday(valentin_ponomorenko, 18-04-1927).
    birthday(dima_ponomorenko, 19-01-1926).
    birthday(ira_ponomorenko, 20-07-1929).
    birthday(maria_ponomorenko, 09-09-1951).
    birthday(igor_rudometkin, 01-07-1950).
    birthday(alexandr_rudometkin, 23-08-1971).

    %%база данных о бракосочитании
    marriage(zhanna, alexandr).
    marriage(alexandr_rudometkin, anastasia_rudometkina).
    mirrage(maria_ponomorenko, igor_rudometkin).
    mirrage(vladimir_p, svetlana_p).
    marriage(g_batargin, g_batargina).
    marriage(andrey_batargin, marina_batargin).
    mirriage(dima_ponomorenko, ira_ponomorenko).
    marriage(valentin_ponomorenko, alla_ponomorenko).
    marriage(vova, irina_ponorenko).
    marriage(lilya, marko).
    %%база о расторжении брака
    unmarriage(artur_lee, irina_lee).
    unmarriage(irina_lee, artur_lee).

    %%база о наличи ребенка
    parent(zhanna, egor).
    parent(alexandr, egor).
    parent(artur_lee, vladimir_lee).
    parent(artur_lee, martin_lee).
    parent(irina_lee, vladimir_lee).
    parent(irina_lee, martin_lee).
    parent(marko, maria-sofia).
    parent(g_batargin, alexandr_batargin).
    parent(g_batargin, andrey_batargin).
    parent(andrey_batargin, ekaterina_batargina).
    parent(marina_batargina, ekaterina_batargina).
    parent(alexandr_rudometkin, evelina_rudometkina).
    parent(anastasia_rudometkina, evelina_rudometkina).
    parent(maria_ponomorenko, alexandr_rudometkin).
    parent(igor_rudometkin, alexandr_rudometkin).
    parent(dima_ponomorenko, maria_ponomorenko).
    parent(dima_ponomorenko, svetlana_p).
    parent(ira_ponomorenko, maria_ponomorenko).

И три правила

    is_relatives(Name1, Name2):-
    parent(X, Name1),
    parent(X, Name2),
    (Name2 \= Name1).
    
    is_grandparent(Name1, Name2):-
        parent(Name1, X),
        parent(X, Name2).
    
    is_cousins(Name1, Name2):- 
       ( parent(X, Name1),
        parent(Y, Name2),
        is_relatives(X, Y));
        (parent(X, Name1),
        parent(Y, Name2),
        is_relatives(X, Z),
        is_cousins(Z, Y)).
Собственно о самой программе main.py

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
        elif patterns[5].replace('имя1', '').replace(" ", '') in req.replace(' ', ''):
            print(UnmarriageParent.find())
        elif patterns[3] in req or patterns[4] in req:
            name1, name2  = input('Введите имена через пробел>> ').split()
            if Unmarriage.find(name1, name2):
                print("Да")
            else:
                print("нет")
        else:
            print('Такого запроса нет')
Здесь есть определнный набор шаблонов с вапросами. Программа не оптимизирована под использование множетсва различных запросов, но в рамках задания раюботае. /
Для работы с Prolog используется библиотека pyswip и функция Prolog. Они позволяют создатьвать запросы к базе знаний записанной в фале Simple.pl
Дальше программа ожидает ввод запроса от пользователя либо q для выхода из программы. После чего иде проверка на то, какой шаблон использован в запросе и, соотвественно, делает переход на нужную функцию. Всего сделано пять файлов с функциями

**FindGrandparents**
FindGrandparents - находит  прародителей человека.

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
Собственно функция prolog.query - делает запрос к базе знаний. И в данном случае используется правило is_grandparent, которое определяет прародителей в Prolog. Результат всегда возвращается в виде массива. 

**FindParent**

Этот кода возвращает список родителей

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

**Unmarriage**

Данная программа позволяет понять, разведен ли или женат ли человек

    from pyswip import Prolog
    
    def find(name1, name2):
        prolog = Prolog
        prolog.consult('Simple.pl')
    
        
        result = []
        for res in prolog.query(f'unmarriage({name1}, {name2}).'):
            result.append(res)
            break
        return len(result) > 0

**UnmirrageParent**

Данная программа не является аналогом правил из Prolog, но его использует. На основе факта unmirrage из Prolog она опеределяет, разведены ли родители ребенка

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
**Cousins**

Оперделяет, являються ли name1 и name2 братьями или сестрами. 

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

# Примеры работы

    Введите запрос. Используйте шаблоны!!!
    Кто родитель Являються ли имя1 и имя2 братьями или сестрами? Кто прародители Разведен Разведена Разведены ли родители ребенка имя1
    Введите запрос(Для выхода нажмите q, а затем enter)>> Кто родитель?

    Введите имя>> egor
    Родителями являються: ['zhanna', 'alexandr']
    Введите запрос(Для выхода нажмите q, а затем enter)>> Являються ли братьями и сестрами?
    Введите имена через пробел еще раз>> anya egor
    Да

    Введите запрос(Для выхода нажмите q, а затем enter)>> Разведен
    Введите имена через пробел>> zhanna alexandr
    нет
    Введите запрос(Для выхода нажмите q, а затем enter)>> Разведены ли родители ребенка?   
    Нет
    
    

    





