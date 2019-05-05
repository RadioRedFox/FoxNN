# FoxNN

# Установка
## Установка для С++
Скачиваем:
activation_function.h, additional_memory.h, foxnn.h, layer.h, neuron.h, settings.h

В проекте добавляем #include "foxnn.h"

На Windows с Visual Studio в настройках проекта С/С++ -> Язык -> Поддержка Open MP -> Да(/openmp)

На Linux в флагах компиляции -std=c++14 -fopenmp

## Установка для Python на Windows с Visual Studio
Дополонительно скачиваем foxnn.i

Скачиваем swig с официального сайта <http://www.swig.org/download.html>. Распаковываем и добавляем папку в Path. 

В командной строке вводим 

swig -c++ -python foxnn.i

Добавляем в проект все скаченные файлы .h и foxnn_wrap.cxx

### В настойках проекта:

С/С++ -> Язык -> Поддержка Open MP -> Да(/openmp)

С/С++ -> Командная строка -> /wd4996 /Zc:twoPhase- 

Общие -> Тип конфигурации -> Динамическая библиотека (.dll)

Общие -> Имя целевого объекта -> _foxnn

Общие -> Конечное расширение -> .pyd

Каталоги VC++ -> Включаемые каталоги -> Python\include

Каталоги VC++ -> Каталог библиотек -> Python\lib

Собираем проект. Переносим в папку с вашим проектом _foxnn.pyd и foxnn.py

В программе пишем import foxnn

## Установка для Python на Linux

Устанавливаем swig

sudo apt install swig

Устанавливаем библиотеки Python

sudo apt-get install python3-dev

Далее собираем саму библиотеку

swig -c++ -python foxnn.i

g++ -c -std=c++14 -fPIC -fopenmp foxnn_wrap.cxx -I/usr/include/python3.6m

g++ -shared foxnn_wrap.o -o _foxnn.so -L /usr/lib/python3.6

Переносим в папку с вашим проектом _foxnn.so и foxnn.py

В программе пишем import foxnn

# Мануал

## Тестовые данные

Нейронную сеть надо тренировать на каких-то данных. Для этого есть класс train_data. 

На с++ train_data data;

На Python data = foxnn.train_data()

Тестовые данные всегда делятся на две части, входные данные и желаемый результат. Для добавления тестовых данных воспользуемся методом add_data

На с++

'''

vector<double> in = { 1, 2, 3}; //на вход нейронная сеть получает три параметра
  
vector<double> out = { 1, 0}; //на выход мы должны получить два значения

data.add_data(in, out);

'''

На Python

'''
data.add_data([1, 2, 3], [0, 1])
'''
