# Линейная регерессия - Калифорния

![](https://media.tenor.com/JVmHSjtpZi8AAAAM/california-roleplay.gif)

# Задание

- Получите и визуализируйте (графически) статистику по датасету (включая количество, среднее значение, стандартное отклонение, минимум, максимум и различные квантили).
- Проведите предварительную обработку данных, включая обработку отсутствующих значений, кодирование категориальных признаков и нормировка.
- Разделите данные на обучающий и тестовый наборы данных.
- Реализуйте линейную регрессию с использованием метода наименьших квадратов без использования сторонних библиотек, кроме NumPy и Pandas (для использования коэффициентов использовать библиотеки тоже нельзя). Использовать минимизацию суммы квадратов разностей между фактическими и предсказанными значениями для нахождения оптимальных коэффициентов.
- Постройте **три модели** с различными наборами признаков.
- Для каждой модели проведите оценку производительности, используя метрику коэффициент детерминации, чтобы измерить, насколько хорошо модель соответствует данным.
- Сравните результаты трех моделей и сделайте выводы о том, какие признаки работают лучше всего для каждой модели.
- Бонусное задание
    - Ввести синтетический признак при построении модели
 
  # Решение



Эта программа представляет собой пример машинного обучения с использованием линейной регрессии для предсказания медианной стоимости домов в Калифорнии на основе набора данных "California Housing". Давайте разберем программу по шагам:

### 1. Импорт библиотек
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```
- **numpy**: Библиотека для работы с массивами и матрицами.
- **pandas**: Библиотека для работы с таблицами данных (DataFrame).
- **matplotlib.pyplot**: Библиотека для визуализации данных.
- **train_test_split**: Функция для разделения данных на обучающую и тестовую выборки.

### 2. Загрузка данных
```python
def load_data():
    data = pd.read_csv("california_housing_train.csv")  # Замените на путь к вашему файлу
    return data
```
- Функция `load_data()` загружает данные из CSV-файла в DataFrame с помощью `pandas`.

### 3. Исследование данных
```python
def explore_data(data):
    print("\nСтатистика по датасету:\n")
    print(data.describe())
    data.hist(figsize=(12, 10))
    plt.show()
```
- Функция `explore_data(data)` выводит статистическую информацию о данных с помощью метода `describe()` и строит гистограммы для всех признаков.

### 4. Предобработка данных
```python
def preprocess_data(data):
    # Обработка пропущенных значений
    data.fillna(data.median(), inplace=True)

    # Нормализация числовых признаков
    numerical_features = data.columns.difference(["median_house_value"])
    data[numerical_features] = (data[numerical_features] - data[numerical_features].mean()) / data[numerical_features].std()

    return data
```
- Функция `preprocess_data(data)` выполняет предобработку данных:
  - Заполняет пропущенные значения медианой.
  - Нормализует числовые признаки (кроме целевой переменной `median_house_value`), вычитая среднее и деля на стандартное отклонение.

### 5. Разделение данных
```python
def split_data(data):
    X = data.drop(columns=["median_house_value"])
    y = data["median_house_value"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
```
- Функция `split_data(data)` разделяет данные на признаки (`X`) и целевую переменную (`y`), а затем разделяет их на обучающую и тестовую выборки в соотношении 80/20.

### 6. Линейная регрессия
```python
def linear_regression(X_train, y_train):
    X = np.c_[np.ones(X_train.shape[0]), X_train]
    y = y_train.values.reshape(-1, 1)
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta.flatten()
```
- Функция `linear_regression(X_train, y_train)` реализует метод наименьших квадратов для линейной регрессии:
  - Добавляет столбец единиц к матрице признаков для учета свободного члена.
  - Вычисляет коэффициенты модели (`theta`) с помощью формулы нормального уравнения.

### 7. Предсказание
```python
def predict(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]
    return X @ theta
```
- Функция `predict(X, theta)` использует обученные коэффициенты (`theta`) для предсказания значений целевой переменной.

### 8. Оценка модели
```python
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)
```
- Функция `r2_score(y_true, y_pred)` вычисляет коэффициент детерминации \( R^2 \), который показывает, насколько хорошо модель объясняет дисперсию целевой переменной.

### 9. Запуск эксперимента
```python
def run_experiment():
    data = load_data()
    explore_data(data)
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    feature_sets = [
        ["median_income"],
        ["total_rooms", "total_bedrooms", "households"],
        ["latitude", "longitude", "housing_median_age", "population"]
    ]

    for i, features in enumerate(feature_sets):
        print(f"\nМодель {i+1} с признаками: {features}")
        theta = linear_regression(X_train[features], y_train)
        y_pred = predict(X_test[features], theta)
        score = r2_score(y_test, y_pred)
        print(f"Коэффициент детерминации (R^2): {score:.4f}")
```
- Функция `run_experiment()` выполняет весь процесс:
  - Загружает данные.
  - Исследует данные.
  - Предобрабатывает данные.
  - Разделяет данные на обучающую и тестовую выборки.
  - Обучает и оценивает модели с разными наборами признаков.

### 10. Запуск программы
```python
if __name__ == "__main__":
    run_experiment()
```
- Этот блок запускает функцию `run_experiment()`, если программа выполняется как основной скрипт.

### Итог
Программа загружает данные о жилье в Калифорнии, исследует их, предобрабатывает, обучает модели линейной регрессии на разных наборах признаков и оценивает их качество с помощью коэффициента детерминации \( R^2 \).
