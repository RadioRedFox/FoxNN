# FoxNN

# Установка
## Установка для С++
Скачиваем:
activation_function.h, additional_memory.h, foxnn.h, layer.h, neuron.h, settings.h  
В проекте добавляем
```cpp
#include "foxnn.h"
```  
На Windows с Visual Studio в настройках проекта С/С++ -> Язык -> Поддержка Open MP -> Да(/openmp)  
На Linux в флагах компиляции -std=c++14 -fopenmp  

## Установка для Python на Windows с Visual Studio
Дополонительно скачиваем foxnn.i  
Скачиваем swig с официального сайта <http://www.swig.org/download.html>. Распаковываем и добавляем папку в Path.  
В командной строке вводим  
```swig -c++ -python foxnn.i```  
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
В программе пишем
```python
import foxnn
```

## Установка для Python на Linux

Устанавливаем swig  
```sudo apt install swig```  
Устанавливаем библиотеки Python  
```sudo apt-get install python3-dev```  
Далее собираем саму библиотеку  
```
swig -c++ -python foxnn.i
g++ -c -std=c++14 -fPIC -fopenmp foxnn_wrap.cxx -I/usr/include/python3.6m
g++ -shared foxnn_wrap.o -o _foxnn.so -L /usr/lib/python3.6
```  
Переносим в папку с вашим проектом _foxnn.so и foxnn.py  
В программе пишем
```python
import foxnn
```

# Мануал

## Тестовые данные

Нейронную сеть надо тренировать на каких-то данных. Для этого есть класс train_data. 

На с++ ```train_data data;```

На Python ```data = foxnn.train_data()```

Тестовые данные всегда делятся на две части, входные данные и желаемый результат. Для добавления тестовых данных воспользуемся методом add_data. Пусть наша нейронная сеть получает на вход три параметра и выдаёт два.

На С++

```cpp

vector<double> in = {1, 2, 3}; //на вход нейронная сеть получает три параметра
  
vector<double> out = {1, 0}; //на выход мы должны получить два значения

data.add_data(in, out);

```

На Python

```python
data.add_data([1, 2, 3], [1, 0]) #на вход два параметра, на выход два параметра
```

Далее можно сохранить в файл, чтобы не повтарять этой процедуры.

```data.save("data.txt")```

Далее можно загружать в программу тестовые данные из этого файла.

```python
data = foxnn.train_data("data.txt")
```

## Создание нейронной сети

Допустим нам надо создать нейронную сеть, получающую на вход 2 параметра, имеющую один скрытый слой с 4 нейронами, 3 нейрона на последнем слое и на выход сеть выдавала бы 2 числа. 

Сеть задаётся вектором, в котором первые n чисел указывают на число нейронов в n-слоях, и n+1-ое число указывает сколько чисел на выход.   
В нашем случае получаем следующее:  
На С++:
```cpp
#include "foxnn.h"
vector <int> nn_parameters = {2, 4, 3, 2};
neural_network nn(nn_parameters);
```
На Python
```
import foxnn
nn = foxnn.neural_network([2, 4, 3, 2])
```

