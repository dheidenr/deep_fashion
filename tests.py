
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('x_train')
print(x_train)
print('y_train')
print(y_train)
print('x_test')
print(x_test)
print('y_test')
print(y_test)

