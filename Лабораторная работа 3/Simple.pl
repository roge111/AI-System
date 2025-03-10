%% demo coming from http://clwww.essex.ac.uk/course/lg519/2-facts/index_18.html
%%
%% please load this file into swi-prolog
%%
%% sam's likes and dislikes in food
%%
%% considering the following will give some practice
%% in thinking about backtracking.
%%
%% you can also run this demo online at
%% http://swish.swi-prolog.org/?code=https://github.com/swi-prolog/swipl-devel/raw/master/demo/likes.pl&q=likes(sam,food).

/** <examples>
?- likes(sam,dahl).
?- likes(sam,chop_suey).
?- likes(sam,pizza).
?- likes(sam,chips).
?- likes(sam,curry).
*/


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
people(martin_lee).
people(svetlana_p).
people(roman).
people(anya).
people(andrey_batargin).
people(ekaterina_batargina).
people(marina_btargina).
people(g_batargin).
people(g_batargina).
people(maria_ponomorenko).
people(igor_rudometkin).
people(alexandr_rudometkin).
people(anastasia_rudometkina).
people(marko).
people(irina_ponomorenko).
people(alla_ponomorenko).
people(valentin_ponomorenko).
people(ira_ponomorenko).
people(dima_ponomorenko).
people(artur_lee).
people(vova).

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
birthday(anastasia_rudometkina, 21-09-1).
birthday(evelina_rudometkina, 24-03-2003).
birthday(valdimir_lee, 11-09-1996).
birthday(martin_lee, 07-09-2013).
birthday(irina_ponomorenko, 15-07-1949).
birthday(vova, 08-02-1948).
birthday(roman, 21-12-1968).
birthday(anya, 05-08-2003).
birthday(lilya, 30-05-1969).
birthday(marko, 27-02-1967).
birthday(mari-sofi, 08-02-2017).

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
parent(ira_ponomorenko, svetlana_p).
parent(svetlana_p, zhanna).
parent(svetlana_p, irina_lee).
parent(vladimir_p, zhanna).
parent(vladimir_p, irina_lee).
parent(alla_ponomorenko, vladimir_p).
parent(alla_ponomorenko, irina_ponomorenko).
parent(valentin_ponomorenko, vladimir_p).
parent(valentin_ponomorenko, irina_ponomorenko).
parent(vova, roma).
parent(vova, lilya).
parent(irina_ponomorenko, roma).
parent(irina_ponomorenko, lilya).
parent(roma, anya).
parent(lilya, mari-sofia).
parent(marko, mari-sofia).



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





