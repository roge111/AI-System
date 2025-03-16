# Анализ данных по Винам

**Задание**
- Проведите предварительную обработку данных, включая обработку отсутствующих значений, кодирование категориальных признаков и масштабирование.
- Получите и визуализируйте (графически) статистику по датасету (включая количество, среднее значение, стандартное отклонение, минимум, максимум и различные квантили), постройте 3d-визуализацию признаков.
- Реализуйте метод k-ближайших соседей ****без использования сторонних библиотек, кроме NumPy и Pandas.
- Постройте две модели k-NN с различными наборами признаков:
    - Модель 1: Признаки случайно отбираются .
    - Модель 2: Фиксированный набор признаков, который выбирается заранее.
- Для каждой модели проведите оценку на тестовом наборе данных при разных значениях k. Выберите несколько различных значений k, например, k=3, k=5, k=10, и т. д. Постройте матрицу ошибок.

# Решение

```df = pd.read_csv('WineDataset.csv')``` - испльзую  библиотеку `pandas`, которую мы импортировали, как `pd`, читаем наш dataset из файла
`print("Пропущенные значения:\n", df.isnull().sum())` - здесь делаем предварительную проверку на пустые значения. В нашем случае их нет. 

Далее делаем масштабирование признаков. Признаки масштабируются с помощью `StandardScaler`, чтобы привести их к единому масштабу (среднее = 0, стандартное отклонение = 1). Это важно для методов, основанных на расстояниях, таких как k-NN.
```
scaler = StandardScaler()
X = df.drop("Wine", axis=1)  # Признаки
y = df["Wine"]  # Целевая переменная
X_scaled = scaler.fit_transform(X)
```

Данные разделяются на обучающую и тестовую выборки в соотношении 80/20:
``` X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) ```

Далее Программа выводит основные статистические характеристики данных (среднее, стандартное отклонение, минимум, максимум и квантили):

``` print("\nОписательная статистика:\n", df.describe()) ```
Строятся гистограммы для каждого признака, чтобы визуально оценить их распределение:
```
df.hist(bins=15, figsize=(15, 10))
plt.suptitle("Распределение признаков")
plt.show()
```

Строится 3D-график для первых трёх признаков (Alcohol, Malic Acid, Ash), чтобы визуализировать их взаимосвязь:

```
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=y, cmap='viridis')
ax.set_xlabel("Alcohol")
ax.set_ylabel("Malic Acid")
ax.set_zlabel("Ash")
plt.title("3D-визуализация признаков")
plt.show()
```

Далее идет реализация метода k-ближайших соседе `k-NN`
Делается это через класс. В данно случае реализоввывает клас `kNN`, который будет выполнять следующие шаги:
 - Вычисляет расстояния между тестовой точкой и всеми точками обучающей выборки.

 - Находит k ближайших соседей.

 - Определяет класс тестовой точки как наиболее часто встречающийся среди k соседей.

```
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train.iloc[k_indices]
            pred = np.bincount(k_nearest_labels).argmax()
            predictions.append(pred)
        return np.array(predictions)
```
Далье построим две модели: случайный и фиксированный наборы признаков:

```
# Модель 1: случайны набор признаков
np.random.seed(42)
random_features = np.random.choice(X.shape[1], size=3, replace=False)
X_train_random = X_train[:, random_features]
X_test_random = X_test[:, random_features]

# Модель 2: фиксированный набор признаков
fixed_features = [0, 1, 2]  # Alcohol, Malic Acid, Ash
X_train_fixed = X_train[:, fixed_features]
X_test_fixed = X_test[:, fixed_features]
```
В модели 2 мы используем три признака: Alcohol, Malic Acid, Ash

Для каждой модели вычисляется точность и строится матрица ошибок при разных значениях k (3, 5, 10):

```
k_values = [3, 5, 10]
for k in k_values:
    print(f"\nОценка для k = {k}")

    # Модель 1: Случайный набор признаков
    knn_random = KNN(k=k)
    knn_random.fit(X_train_random, y_train)
    y_pred_random = knn_random.predict(X_test_random)
    print(f"Точность модели 1 (случайные признаки): {accuracy_score(y_test, y_pred_random):.2f}")
    print("Матрица ошибок модели 1:\n", confusion_matrix(y_test, y_pred_random))

    # Модель 2: Фиксированный набор признаков
    knn_fixed = KNN(k=k)
    knn_fixed.fit(X_train_fixed, y_train)
    y_pred_fixed = knn_fixed.predict(X_test_fixed)
    print(f"Точность модели 2 (фиксированные признаки): {accuracy_score(y_test, y_pred_fixed):.2f}")
    print("Матрица ошибок модели 2:\n", confusion_matrix(y_test, y_pred_fixed))
```

# Итог

Все результаты можно наблюдать в фале ![Analitic.ipynb](https://github.com/roge111/AI-System/blob/main/Modul%202/Analitic.ipynb). Там приведены графики и все выводы программ.

Ну а что у нас в итоге выполняет программа? 

 - Загружает и предварительно обрабатывает данные
 - Визулизирует обработанные даные
 - Реализует метод `k-NN`
 - Сравнивает две модели `k-NN` с разными наборами признаков
 - Оценивает модели на тестовых данных и выводит резульаты


# Вывод
В ходе выполнения задания был изучен метод k-ближайших соседей -  это один из самых простыз и интуитивно понятных алгоритмов машинного обучения.

