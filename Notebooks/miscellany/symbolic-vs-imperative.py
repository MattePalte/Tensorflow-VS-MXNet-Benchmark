# Symbolic ( or Declarative)
from keras.models import Sequential
from keras.layers import Dense, Activation

# Define the graph
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# Compile the graph
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Imperative
import numpy as np

a = np.ones(10)
b = np.ones(10) * 2
c = b * a
d = c + 1

# Symbolic
A = Variable('A')
B = Variable('B')
C = B * A
D = C + Constant(1)
# compiles the function
f = compile(D)
d = f(A=np.ones(10), B=np.ones(10)*2)



