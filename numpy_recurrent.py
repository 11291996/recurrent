import numpy as np
import sys 
sys.path.append('../')

from activation import activation

activation = activation()

np.random.seed(0)

w = np.random.rand(9, 2)

x1 = np.array([1,2,3])
x2 = np.array([4,5,6])
x3 = np.array([7,8,9])

e1 = np.concatenate((x1, x2, x3), axis=0)
e2 = np.concatenate((x2, x1, x3), axis=0)

e = np.array([e1, e2])

vocabulary = ('apple', 'bannana', 'cake')

sentence = [0 for i in range(3)]

for _ in range(3):
    sentence[_] = np.array(e1[_:_ + 3])

y = np.zeros((1, len(vocabulary)))
for i in range(len(sentence)):
    w = np.random.rand(len(vocabulary), len(vocabulary))
    y += w@sentence[i]
print(activation.softmax(y, std = False))

for _ in range(3):
    sentence[_] = np.array(e2[_:_ + 3])

y = np.zeros((1, len(vocabulary)))
for i in range(len(sentence)):
    w = np.random.rand(len(vocabulary), len(vocabulary))
    y += w@sentence[i]

print(activation.sigmoid(np.sum(x1)))