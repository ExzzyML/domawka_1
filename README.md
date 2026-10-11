# Линейные модели с нуля

Линейная и логистическая регрессии на `numpy` с собственным градиентным спуском и аналитическим решением. Внутри пакета `linear_models` библиотека `scikit-learn` не используется; она нужна только ноутбуку `experiments.ipynb`, где свои модели сравниваются с `sklearn`.

Проверено на Python 3.12.7, `numpy` 2.5.3, `scikit-learn` 1.9.1.

## Установка и быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; на Linux и macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`linear_models/__init__.py` пустой, поэтому классы импортируются из своих модулей:

```python
import numpy as np
from linear_models.linear_regression import LinearRegression
from linear_models.logistic_regression import LogisticRegression

# Регрессия: точная прямая y = 2x + 1, аналитическое решение
X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = 2 * X[:, 0] + 1
model = LinearRegression(solver="analytic").fit(X, y)
print(model.coef_, model.intercept_)          # [2.] 1.0

# Та же задача градиентным спуском с L2-штрафом (веса «сжимаются»)
model = LinearRegression(
    solver="gradient", penalty="l2", alpha=0.1,
    learning_rate=0.1, max_iter=5000, tol=1e-9,
).fit(X, y)
print(model.coef_.round(3), round(model.intercept_, 3), model.n_iter_)   # [1.852] 1.222 219

# Классификация
X = np.array([[-3.0], [-1.0], [1.0], [2.0]])
y = np.array([0, 0, 1, 1])
clf = LogisticRegression(penalty="l2", alpha=0.01, learning_rate=0.5, max_iter=5000, tol=1e-9).fit(X, y)
print(clf.predict(X))                         # [0 0 1 1]
print(clf.predict_proba(X).round(3))          # столбцы: P(класс 0), P(класс 1)
```

## Архитектура

```text
BaseLinearModel (абстрактный)
├── LinearRegression
└── LogisticRegression
```

- `BaseLinearModel` (`linear_models/base.py`) хранит параметры, проверяет их и входные данные, содержит общий цикл градиентного спуска `_fit_gradient`, штраф и его градиент, ранний останов, `coef_`, `intercept_`, `n_iter_`, `loss_history_`. Подклассы реализуют `fit`, `predict`, `_loss` и `_gradient`.
- `LinearRegression` (`linear_models/linear_regression.py`): потери `1/(2n) · Σ(z − y)²`; два solver'а.
- `LogisticRegression` (`linear_models/logistic_regression.py`): потери `mean(log(1 + e^z) − y·z)`, посчитанные через `np.logaddexp`; устойчивая сигмоида; есть `predict_proba` и порог `threshold`.
- `NotFittedError` (наследник `ValueError`) бросается, если вызвать `predict` до `fit`.

### Параметры конструктора

| Параметр | По умолчанию | Смысл |
|---|---|---|
| `fit_intercept` | `True` | обучать свободный член; при `False` `intercept_ = 0.0` |
| `solver` | `"gradient"` | `"gradient"` или `"analytic"` (только у `LinearRegression`) |
| `penalty` | `None` | `None`, `"l1"` или `"l2"` |
| `alpha` | `0.0` | сила штрафа, число `>= 0` |
| `learning_rate` | `0.01` | шаг градиентного спуска, число `> 0` |
| `max_iter` | `1000` | максимум итераций, целое `>= 1` |
| `tol` | `1e-6` | порог раннего останова, число `>= 0` |
| `verbose`, `random_state` | `False`, `None` | принимаются для совместимости с интерфейсом, ни на что не влияют (см. «Ограничения») |
| `threshold` | `0.5` | только `LogisticRegression`: класс 1, если `P(1) > threshold` |

### Атрибуты после `fit`

| Атрибут | Что это |
|---|---|
| `coef_` | веса, массив формы `(n_features,)` |
| `intercept_` | свободный член (число) |
| `n_iter_` | число выполненных итераций (у `analytic` равно 1) |
| `loss_history_` | список значений полной цели `J` после каждой итерации |

`predict(X)` у регрессии возвращает значения `z = X @ coef_ + intercept_`; у классификации возвращает классы `0` и `1` (тип `int`), `predict_proba(X)` возвращает массив формы `(n, 2)`.

## Solver'ы

| Solver | Где | Что делает |
|---|---|---|
| `"analytic"` | `LinearRegression`, `penalty` только `None` или `"l2"` | центрирует данные (если `fit_intercept=True`) и решает задачу наименьших квадратов через `np.linalg.lstsq`; L2 добавляет строки `√(n·alpha)·I` |
| `"gradient"` | обе модели, любой `penalty` | градиентный спуск с постоянным шагом `learning_rate`; веса стартуют с нулей |

`LogisticRegression` поддерживает только `"gradient"`, у неё нет замкнутой формы.

## Регуляризация

Цель, которую минимизируют обе модели: `J(w, b) = потери(w, b) + alpha · R(w)`, где потери усреднены по объектам.

| `penalty` | `R(w)` | Градиент штрафа |
|---|---|---|
| `None` | 0 | нет |
| `"l2"` | `½ · ‖w‖²` | `alpha · w` |
| `"l1"` | `Σ\|w_j\|` | `alpha · sign(w)` (субградиент) |

Свободный член `intercept_` не штрафуется ни в одном режиме.

## Early stopping

После каждого обновления весов считается полная цель `J` (потери плюс штраф) и дописывается в `loss_history_`. Спуск останавливается, как только `|J_{k-1} − J_k| < tol`. Поэтому:

- `n_iter_ == len(loss_history_)`;
- `tol = 0.0` отключает остановку: выполнится ровно `max_iter` итераций;
- правило смотрит на изменение цели, а не на расстояние до решения: на плохо обусловленных данных оно срабатывает рано (см. раздел 5.3 ноутбука).

## Соответствие параметрам `sklearn`

Наша цель делится на `n`, поэтому `alpha` нужно пересчитывать (здесь `n` — число обучающих объектов):

| Наша модель | Модель sklearn | Соответствие |
|---|---|---|
| `LinearRegression`, `penalty="l2"`, `alpha` | `Ridge` | `alpha_sklearn = n · alpha` |
| `LinearRegression`, `penalty="l1"`, `alpha` | `Lasso` | `alpha_sklearn = alpha` |
| `LogisticRegression`, `penalty="l2"`, `alpha` | `LogisticRegression(l1_ratio=0, C=C)` | `C = 1 / (n · alpha)` |
| `LogisticRegression`, `penalty="l1"`, `alpha` | `LogisticRegression(l1_ratio=1, solver="saga", C=C)` | `C = 1 / (n · alpha)` |
| без штрафа | `LogisticRegression(C=np.inf)` | параметр `penalty` в sklearn 1.9 устарел |

## Что показал ноутбук

Те же данные (`train_test_split`, `random_state=42`, стандартизация по обучающей части) подавались и в свои модели, и в `sklearn`:

- линейная регрессия (analytic и gradient, без штрафа и с L2): коэффициенты совпадают с `sklearn` до `6.4e-13`;
- логистическая регрессия (без штрафа и с L2): коэффициенты до `2.6e-7`, классы на тесте совпадают на 100 % (кроме разделимого случая ниже); разность задаётся точностью самого `lbfgs` в sklearn, а не нашей;
- L1: значение цели совпадает с `sklearn`, но точных нулей у нас нет (колебания порядка `learning_rate · alpha`), у `saga` они есть;
- на линейно разделимых данных без штрафа (`breast_cancer`) веса растут вместе с числом итераций, поэтому сравниваются предсказания, а не коэффициенты;
- спуск с постоянным шагом сходится, только если `learning_rate < 2/λmax`; на нестандартизованных признаках это ограничение в тысячи раз жёстче, поэтому признаки перед обучением нужно стандартизовать.

## Ограничения

- Остановка по абсолютному изменению цели (`tol`), а не по норме градиента и не по расстоянию до решения.
- L1 решается субградиентным спуском: коэффициенты малы, но точных нулей нет.
- `random_state` и `verbose` принимаются, но не используются: веса всегда стартуют с нулей, вывод в процессе обучения не печатается.
- Слишком большой `learning_rate` приводит к расходимости: `J` становится `inf` или `nan`, остановка не срабатывает.
- На линейно разделимых данных без штрафа оптимума нет, веса растут до остановки по `tol` или `max_iter`.
- У `analytic` вырожденная матрица (линейно зависимые признаки) даёт решение минимальной нормы без предупреждения.
- `NaN` и `inf` в `X` и `y` не проверяются: `analytic` может завершиться ошибкой `LinAlgError`, а `gradient` вернёт веса `nan`.
- `P(0)` считается как `1 − P(1)`, поэтому в далёком хвосте теряет точность (на классы не влияет); класс 1 присваивается при строгом `P(1) > threshold`.
- Нет `sample_weight`, мини-батчей, шага, зависящего от итерации, и многоклассовой классификации.

## Запуск тестов

```bash
pytest
```

Настройки лежат в `pytest.ini` (`testpaths = tests`, `pythonpath = .`). Сейчас 157 тестов: параметры и входные данные (`tests/test_validation.py`), линейная (`tests/test_linear_regression.py`) и логистическая регрессии (`tests/test_logistic_regression.py`).

## Запуск ноутбука

1. Установите зависимости: `pip install -r requirements.txt` (в ту же среду `.venv`).
2. Откройте `experiments.ipynb` в VS Code или выполните `jupyter notebook experiments.ipynb`, выберите ядро из `.venv`.
3. Выполните Kernel → Restart & Run All. Все ячейки идут подряд и не требуют ручных правок; полный прогон занимает от десятков секунд до нескольких минут. Версии библиотек печатаются в первой ячейке.

В репозитории ноутбук лежит с сохранёнными выводами и графиками, поэтому его можно читать без запуска. Значения в строках с сырыми нестандартизованными признаками (раздел 5.2, большие `learning_rate`) зависят от машины: там цель хаотически скачет.
