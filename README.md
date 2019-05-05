# FoxNN

# Установка
## Установка для С++
Скачиваем:
activation_function.h, additional_memory.h, foxnn.h, layer.h, neuron.h, settings.h

В проекте добавляем #include "foxnn.h"

## Установка для Python на Windows с Visual Studio
Дополонительно скачиваем foxnn.i

Скачиваем swig с официального сайта http://www.swig.org/download.html . Распаковываем и добавляем папку в Path. 

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
