# FoxNN

Данная библиоотека преставляет собой инструмент для быстрого и очень простого создания нейронной сети.

Инструкцию по установке и использованию вы можете найти [здесь](https://github.com/RadioRedFox/FoxNN/wiki).

Библиотека работает как на Windows так и на Linux. Есть api для испльзования библиотеки в Python. 


### Простая идея.
[Сеть создаётся](https://github.com/RadioRedFox/FoxNN/wiki/3.-%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%BE%D0%B9-%D1%81%D0%B5%D1%82%D0%B8.) одной строчкой кода с указанием числа нейронов в слоях и сколько целевых значений.
```python
nn = foxnn.neural_network([5, 10, 3, 2]) # три слоя и 2 числа на выходе
```
Легко и понятно работать с [обучающей выборкой](https://github.com/RadioRedFox/FoxNN/wiki/2.-%D0%A0%D0%B0%D0%B1%D0%BE%D1%82%D0%B0-%D1%81-%D0%BE%D0%B1%D1%83%D1%87%D0%B0%D1%8E%D1%89%D0%B5%D0%B9-%D0%B2%D1%8B%D0%B1%D0%BE%D1%80%D0%BA%D0%BE%D0%B9.). Первый массив входные данные, второй массив целевые значения.  

```python
# создаём тренировачные данные
train_data = foxnn.train_data()
# добавили 1 тест, [1, 2, 3] - входные данные, [1, 0] - то, что хотим получить на выход
train_data.add_data([1, 2, 3], [1, 0]) 
```

[Запуск обучения](https://github.com/RadioRedFox/FoxNN/wiki/6.-%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5-%D1%81%D0%B5%D1%82%D0%B8.)
```python
nn.train_on_data(data_for_train=train_data, speed=0.01, max_iteration=100, size_train_batch=98)
```

Получить выходное значение сети
```python
nn.get_out([0, 1, 2, 2, 0.1])
```

Есть оптимизации [Адама](https://github.com/RadioRedFox/FoxNN/wiki/7.-%D0%9E%D0%BF%D1%82%D0%B8%D0%BC%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F.#%D0%9E%D0%BF%D1%82%D0%B8%D0%BC%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F-%D0%90%D0%B4%D0%B0%D0%BC%D0%B0) и [Нестерова](https://github.com/RadioRedFox/FoxNN/wiki/7.-%D0%9E%D0%BF%D1%82%D0%B8%D0%BC%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F.#%D0%9E%D0%BF%D1%82%D0%B8%D0%BC%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F-%D0%9D%D0%B5%D1%81%D1%82%D0%B5%D1%80%D0%BE%D0%B2%D0%B0). 
```python
nn.settings.settings_optimization.set_mode("Adam")
nn.settings.settings_optimization.set_mode("Nesterov")
```

Есть возможность [рандомного изменения весов](https://github.com/RadioRedFox/FoxNN/wiki/8.-%D0%9C%D1%83%D1%82%D0%B0%D1%86%D0%B8%D1%8F) для моделирования эволюции:
```python
nn.random_mutation(0.1) #  рандомное изменение весов
nn.smart_mutation(0.1)  #  изменение весов на величину соизмеримую с исходим значением весов
```
