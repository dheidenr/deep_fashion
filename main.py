from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

# Загрузка данных из сети
# x_train - изображения, которые будут использоваться для обучения
# y_train - правильные ответы (метки классов, которые говорят какой именно объект указан на изображении)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Преобразование размерности изображений. Преобразуем изображения в вектор
x_train = x_train.reshape(60000, 784)

# Нормализация данных делится интенсивность каждого пикселя на 255,
# таким образом получиться что данные на входе в сеть будут от нуля до единицы, что удобно для обучения нейросети
# x_backup_train = x_train
# x_train = x_test.copy
# x_train /= 255


# Преобразуем метки в категаории
y_train = utils.to_categorical(y_train, 10)

# Названия классов
classes = ['футболка', "брюки", "свитер", "платье", "пальто", "туфли", "рубашка", "крассовки", "сумка", "ботинки"]

# Создаем последовательную модель
model = Sequential()

# Добавляем уровни сети
# Входной слой
model.add(Dense(800, input_dim=784, activation="relu"))
# 10 - количество нейронов
model.add(Dense(10, activation="softmax"))

# компиляция модели. Тип оптимизатора: SGD - стахостический градиентный спуск, categorical_crossentropy - фукнция ошибки
# которая хорошо подюходит для задач в которых количество элементов классификации больше двух
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)

print('Hello deep fashion')
