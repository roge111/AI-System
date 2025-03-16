# Дерево решений - анализ данных по грибам

![](https://camo.githubusercontent.com/dfb55b22f37809e5daf0e86d1aa84f7655a97998d95d0daab362896232fc813e/68747470733a2f2f63646e2e6472696262626c652e636f6d2f75736572732f3633343530382f73637265656e73686f74732f323137323038332f6d656469612f38363364613836656561656430353634343462653466633862303265646364662e676966)

# Требуется
1. Отобрать **случайным** образом sqrt(n) признаков
2. Реализовать без использования сторонних библиотек построение дерева решений  (дерево не бинарное, numpy и pandas использовать можно, использовать список списков  для реализации  дерева - нельзя) для решения задачи бинарной классификации 
3. Провести оценку реализованного алгоритма с использованием Accuracy, precision и recall
4. Построить кривые AUC-ROC и AUC-PR (в пунктах 4 и 5 использовать библиотеки нельзя)

# Решение
 Однако тут проблема в том, что нет прямого файла, поэтому мы dataset импортируем из вне программы

 За загрузку отвечает это код:
```
# Установим пакет ucimlrepo
!pip install ucimlrepo

# Импортируем необходимые библиотеки
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Загружаем датасет
mushroom = fetch_ucirepo(id=73)

# Данные
X = mushroom.data.features  # Признаки
y = mushroom.data.targets   # Целевая переменная

# Преобразуем целевую переменную в бинарный формат (0 и 1)
y = y.apply(lambda x: 1 if x == 'p' else 0)
```

Теперь у нас `x` — признаки, а `y` — целевая переменная, которая была преобразована в бинарный формат, где ядовитый гриб = 1, а съедобный гриб = 0. Это упростит дальнейшую работу с алгоритмами.


Производим отбор случайных признаков. Это делается через функцию `random` из библиотеки `numpy`, которая испортирована как `np`
```
# Вычисляем количество признаков для отбора
n_features = int(np.sqrt(X.shape[1]))

# Случайным образом отбираем признаки
selected_features = np.random.choice(X.columns, size=n_features, replace=False)
X_selected = X[selected_features]
```

# Реализация дерева решений

Реализация дерева решений — это процесс создания модели, которая рекурсивно разделяет данные на подмножества на основе значений признаков, чтобы максимизировать "чистоту" целевой переменной в каждом подмножестве. Давайте разберем этот процесс подробно.

---

### Основные шаги реализации дерева решений

1. **Выбор критерия для разделения**:
   - Мы используем критерий, такой как **индекс Джини** или **энтропия**, чтобы определить, насколько хорошо разделение данных улучшает "чистоту" целевой переменной.
   - В нашем примере будем использовать **индекс Джини**.

2. **Поиск лучшего разделения**:
   - Для каждого признака и каждого возможного значения этого признака мы вычисляем, насколько хорошо разделение данных улучшает критерий (например, уменьшает индекс Джини).

3. **Рекурсивное построение дерева**:
   - После нахождения лучшего разделения мы рекурсивно применяем тот же процесс к каждому подмножеству данных.

4. **Критерии остановки**:
   - Рекурсия останавливается, если:
     - Глубина дерева достигает максимального значения.
     - Количество образцов в узле становится меньше минимального значения.
     - Все образцы в узле принадлежат одному классу.

5. **Предсказание**:
   - Для нового образца мы проходим по дереву от корня до листа, используя значения признаков, и возвращаем класс, соответствующий листу.

---

### Реализация дерева решений на Python

Давайте подробно разберем реализацию класса `DecisionTree`, который представляет собой алгоритм построения дерева решений для задачи классификации. Этот класс реализует рекурсивное разделение данных на основе индекса Джини и позволяет предсказывать классы для новых данных.

---

### Структура класса `DecisionTree`

#### 1. **Инициализация (`__init__`)**:
```python
def __init__(self, max_depth=5, min_samples_split=2):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.tree = None
```
- **`max_depth`**: Максимальная глубина дерева. Ограничивает глубину рекурсии.
- **`min_samples_split`**: Минимальное количество образцов, необходимое для дальнейшего разделения узла.
- **`tree`**: Структура дерева, которая будет построена в процессе обучения.

---

#### 2. **Критерий Джини (`gini`)**:
```python
def gini(self, y):
    """Вычисляет индекс Джини для целевой переменной."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)
```
- **Индекс Джини** измеряет "нечистоту" узла. Чем меньше значение, тем чище узел (все образцы принадлежат одному классу).
- Формула: \( \text{Gini} = 1 - \sum_{i=1}^C p_i^2 \), где \(C\) — количество классов, а \(p_i\) — доля образцов класса \(i\).

---

#### 3. **Разделение данных (`split`)**:
```python
def split(self, X, y, feature, threshold):
    """Разделяет данные по признаку и пороговому значению."""
    left_mask = X[feature] <= threshold
    right_mask = X[feature] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
```
- Разделяет данные на два подмножества:
  - Левое подмножество: образцы, где значение признака `feature` меньше или равно `threshold`.
  - Правое подмножество: образцы, где значение признака `feature` больше `threshold`.

---

#### 4. **Поиск лучшего разделения (`find_best_split`)**:
```python
def find_best_split(self, X, y):
    """Находит лучшее разделение данных."""
    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    for feature in X.columns:
        thresholds = np.unique(X[feature])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            gini = (len(y_left) * self.gini(y_left) + len(y_right) * self.gini(y_right)) / len(y)
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold
```
- Для каждого признака и каждого уникального значения этого признака:
  - Вычисляется индекс Джини после разделения.
  - Выбирается разделение с минимальным значением индекса Джини.
- Возвращается лучший признак и пороговое значение для разделения.

---

#### 5. **Рекурсивное построение дерева (`build_tree`)**:
```python
def build_tree(self, X, y, depth=0):
    """Рекурсивно строит дерево решений."""
    if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
        return np.argmax(np.bincount(y))

    feature, threshold = self.find_best_split(X, y)
    if feature is None:
        return np.argmax(np.bincount(y))

    X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
    left_subtree = self.build_tree(X_left, y_left, depth + 1)
    right_subtree = self.build_tree(X_right, y_right, depth + 1)

    return {'feature': feature, 'threshold': threshold,
            'left': left_subtree, 'right': right_subtree}
```
- Рекурсивно строит дерево:
  - Если достигнута максимальная глубина, или в узле слишком мало образцов, или все образцы принадлежат одному классу, возвращается класс с наибольшим количеством образцов.
  - Иначе:
    - Находится лучшее разделение.
    - Рекурсивно строятся левое и правое поддеревья.
- Возвращается структура дерева в виде словаря:
  - `feature`: Признак для разделения.
  - `threshold`: Пороговое значение.
  - `left`: Левое поддерево.
  - `right`: Правое поддерево.

---

#### 6. **Обучение модели (`fit`)**:
```python
def fit(self, X, y):
    """Обучает дерево решений."""
    self.tree = self.build_tree(X, y)
```
- Вызывает метод `build_tree` для построения дерева на основе данных `X` и `y`.

---

#### 7. **Предсказание для одного образца (`predict_sample`)**:
```python
def predict_sample(self, sample, tree):
    """Предсказывает класс для одного образца."""
    if not isinstance(tree, dict):
        return tree
    feature = tree['feature']
    threshold = tree['threshold']
    if sample[feature] <= threshold:
        return self.predict_sample(sample, tree['left'])
    else:
        return self.predict_sample(sample, tree['right'])
```
- Рекурсивно проходит по дереву:
  - Если текущий узел — лист (не словарь), возвращает класс.
  - Иначе решает, в какое поддерево перейти, на основе значения признака и порога.

---

#### 8. **Предсказание для всех образцов (`predict`)**:
```python
def predict(self, X):
    """Предсказывает класс для всех образцов."""
    return np.array([self.predict_sample(row, self.tree) for _, row in X.iterrows()])
```
- Применяет `predict_sample` ко всем строкам в `X` и возвращает массив предсказанных классов.

---

### Пример использования

```python
# Создаем экземпляр дерева
tree = DecisionTree(max_depth=5, min_samples_split=2)

# Обучаем модель
tree.fit(X_train, y_train)

# Предсказываем классы
y_pred = tree.predict(X_test)

# Оцениваем точность
accuracy = np.mean(y_test == y_pred)
print("Accuracy:", accuracy)
```

---

### Преимущества реализации
1. **Простота**: Код легко понять и модифицировать.
2. **Гибкость**: Можно настроить параметры, такие как `max_depth` и `min_samples_split`.
3. **Рекурсивность**: Рекурсивный подход делает код компактным и читаемым.

---

### Ограничения
1. **Скорость**: Реализация может быть медленной для больших наборов данных.
2. **Переобучение**: Без ограничения глубины дерево может переобучиться.
3. **Категориальные признаки**: Реализация не поддерживает категориальные признаки напрямую (их нужно предварительно закодировать).

---

Эта реализация является базовой и может быть улучшена, например, добавлением поддержки категориальных признаков, оптимизацией скорости или реализацией других критериев (например, энтропии).

# Оценка алгоритма

Требуется оценивать алгоритм без испльзований биоблитек. Но поскольку у нас все данные по-умолчанию храняться в формате `numpy`, то будет использовать минимальные функции `numpy`:

- `np.sum()` - суммирует то, что находиться в скобках
- `np.mean()` - находит среднее значение
- `DecisionTree`  - класс, который был реализован прямо в программе
 Для `sum` и `mean` есть аналоги из базового набора функций Python, которые применимы к спискам, поэтому нет разницы, что в `numpy` формате, что в обычном.

Перейдем к программе:
```
#Оценка алгоритма
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

# Обучаем дерево
tree = DecisionTree(max_depth=5)
tree.fit(X_selected, y)

# Предсказываем
y_pred = tree.predict(X_selected)

# Оцениваем
print("Accuracy:", accuracy(y, y_pred))
print("Precision:", precision(y, y_pred))
print("Recall:", recall(y, y_pred))
```

Разберем подробно. Этот код выполняет обучение модели дерева решений, предсказание на данных и оценку качества модели с помощью метрик **Accuracy**, **Precision** и **Recall**. Давайте разберем его по частям.

---

### 1. **Функции для оценки алгоритма**

#### a) **Accuracy (Точность)**:
```python
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
```
- **Что делает**:
  - Вычисляет долю правильных предсказаний относительно общего числа предсказаний.
- **Формула**:
  \[
  \text{Accuracy} = \frac{\text{Количество правильных предсказаний}}{\text{Общее количество предсказаний}}
  \]
- **Пример**:
  - Если `y_true = [1, 0, 1, 0]` и `y_pred = [1, 0, 0, 0]`, то:
    \[
    \text{Accuracy} = \frac{2}{4} = 0.5
    \]

#### b) **Precision (Точность)**:
```python
def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0
```
- **Что делает**:
  - Вычисляет долю правильно предсказанных положительных классов (`1`) относительно всех предсказанных положительных классов.
- **Формула**:
  \[
  \text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
  \]
- **Пример**:
  - Если `y_true = [1, 0, 1, 0]` и `y_pred = [1, 0, 0, 1]`, то:
    \[
    \text{Precision} = \frac{1}{2} = 0.5
    \]

#### c) **Recall (Полнота)**:
```python
def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0
```
- **Что делает**:
  - Вычисляет долю правильно предсказанных положительных классов (`1`) относительно всех фактических положительных классов.
- **Формула**:
  \[
  \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  \]
- **Пример**:
  - Если `y_true = [1, 0, 1, 0]` и `y_pred = [1, 0, 0, 0]`, то:
    \[
    \text{Recall} = \frac{1}{2} = 0.5
    \]

---

### 2. **Обучение модели**

```python
tree = DecisionTree(max_depth=5)
tree.fit(X_selected, y)
```
- **Что делает**:
  - Создает экземпляр модели дерева решений с максимальной глубиной `5`.
  - Обучает модель на данных `X_selected` (признаки) и `y` (целевая переменная).

---

### 3. **Предсказание**

```python
y_pred = tree.predict(X_selected)
```
- **Что делает**:
  - Использует обученную модель для предсказания классов на тех же данных (`X_selected`).
  - Возвращает массив предсказанных классов `y_pred`.

---

### 4. **Оценка модели**

```python
print("Accuracy:", accuracy(y, y_pred))
print("Precision:", precision(y, y_pred))
print("Recall:", recall(y, y_pred))
```
- **Что делает**:
  - Вычисляет и выводит значения метрик **Accuracy**, **Precision** и **Recall**.
  - Сравнивает истинные значения `y` с предсказанными `y_pred`.

# Кривые AUC-ROC и AUC-PR

Этот код строит кривые **AUC-ROC** и **AUC-PR** для оценки качества модели классификации. Давайте разберем его по частям.

---

### 1. **Кривая ROC (Receiver Operating Characteristic)**

#### a) **Функция `roc_curve`**:
```python
def roc_curve(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr = []
    fpr = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))

        tpr.append(true_positives / (true_positives + false_negatives))
        fpr.append(false_positives / (false_positives + true_negatives))

    return fpr, tpr, thresholds
```

- **Что делает**:
  - Вычисляет значения **True Positive Rate (TPR)** и **False Positive Rate (FPR)** для различных пороговых значений.
  - **TPR** (True Positive Rate) — это доля правильно предсказанных положительных классов (Recall).
  - **FPR** (False Positive Rate) — это доля отрицательных классов, ошибочно предсказанных как положительные.

- **Параметры**:
  - `y_true`: Истинные значения классов (0 или 1).
  - `y_scores`: Предсказанные вероятности или "уверенность" модели в принадлежности к классу 1.

- **Алгоритм**:
  1. Упорядочивает уникальные значения `y_scores` по убыванию (пороговые значения).
  2. Для каждого порога:
     - Преобразует `y_scores` в бинарные предсказания (`1`, если `y_scores >= threshold`, иначе `0`).
     - Вычисляет **TPR** и **FPR**.
  3. Возвращает списки **FPR**, **TPR** и пороговые значения.

---

#### b) **Пример использования**:
```python
fpr, tpr, _ = roc_curve(y, y_pred)
```

- **Что делает**:
  - Вычисляет значения **FPR** и **TPR** для построения кривой ROC.

---

### 2. **Кривая PR (Precision-Recall)**

#### a) **Функция `pr_curve`**:
```python
def pr_curve(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    precision = []
    recall = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        actual_positives = np.sum(y_true == 1)

        precision.append(true_positives / predicted_positives if predicted_positives > 0 else 0)
        recall.append(true_positives / actual_positives if actual_positives > 0 else 0)

    return recall, precision, thresholds
```

- **Что делает**:
  - Вычисляет значения **Precision** и **Recall** для различных пороговых значений.
  - **Precision** — это доля правильно предсказанных положительных классов относительно всех предсказанных положительных классов.
  - **Recall** — это доля правильно предсказанных положительных классов относительно всех фактических положительных классов.

- **Параметры**:
  - `y_true`: Истинные значения классов (0 или 1).
  - `y_scores`: Предсказанные вероятности или "уверенность" модели в принадлежности к классу 1.

- **Алгоритм**:
  1. Упорядочивает уникальные значения `y_scores` по убыванию (пороговые значения).
  2. Для каждого порога:
     - Преобразует `y_scores` в бинарные предсказания (`1`, если `y_scores >= threshold`, иначе `0`).
     - Вычисляет **Precision** и **Recall**.
  3. Возвращает списки **Recall**, **Precision** и пороговые значения.

---

#### b) **Пример использования**:
```python
recall_pr, precision_pr, _ = pr_curve(y, y_pred)
```

- **Что делает**:
  - Вычисляет значения **Recall** и **Precision** для построения кривой PR.

---

### 3. **Визуализация кривых**

#### a) **Кривая ROC**:
```python
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC Curve")
```

- **Что делает**:
  - Строит график зависимости **TPR** от **FPR**.
  - Идеальная модель имеет кривую, проходящую через верхний левый угол (FPR = 0, TPR = 1).

---

#### b) **Кривая PR**:
```python
plt.subplot(1, 2, 2)
plt.plot(recall_pr, precision_pr, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("AUC-PR Curve")
```

- **Что делает**:
  - Строит график зависимости **Precision** от **Recall**.
  - Идеальная модель имеет кривую, проходящую через верхний правый угол (Recall = 1, Precision = 1).

---

### 4. **Итоговый график**

```python
plt.show()
```

- **Что делает**:
  - Отображает оба графика (ROC и PR) в одном окне.

# Вывод

В ходе работы была выполнена реализация деерва решений. Я считаю для себя это очень полезно, поскольку стремлюсь развиваться в аналитеке данных.



