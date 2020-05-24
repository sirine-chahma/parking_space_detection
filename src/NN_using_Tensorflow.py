#Algorithme de réseau de neuronne fully connected sans utilisation de Keras

from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


X = tf.placeholder(tf.float32, [None, 28, 28, 1])  #Matrice contenant un lot d'images
W = tf.Variable(tf.zeros([28 * 28, 2])) #Matrice des poids
b = tf.Variable(tf.zeros([2]))  #Matrice des biais

init = tf.global_variables_initializer()

# modèle
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 28 * 28]), W) + b)
# Matrice pour les valeurs correctes
Y_ = tf.placeholder(tf.float32, [None, 2])

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % de réponses correctes dans un lot d'images
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#Optimisation par descente de gradient
optimizer = tf.train.GradientDescentOptimizer(0.0003)
train_step = optimizer.minimize(cross_entropy)

#Lancement de la session
sess = tf.Session()
sess.run(init)

#Chargement des images de la base de données
pickle_in = open("X.pickle", "rb")
Xpickle = pickle.load(pickle_in)

#Chargement des labels de la base de données
pickle_in = open("y.pickle", "rb")
Ypickle_init = pickle.load(pickle_in)

#Modification des labels pour que une place busy ne soit plus 0 mais [1,0] et une place free ne soit plus 1 mais [0,1]

def modif_Y(Ypickle_init):
    Ypickle = []
    for i in range(len(Ypickle_init)):
        if Ypickle_init[i] == 0:
            Ypickle.append([1, 0])
        if Ypickle_init[i] == 1:
            Ypickle.append([0, 1])
    return Ypickle

# Compilation du modèle

A = []
index = []
for j in range(1):
    train_data = {X: Xpickle[j: j+100], Y_: modif_Y(Ypickle_init)[j: j+100]}
    _, loss_val, a = sess.run([train_step, cross_entropy, accuracy], feed_dict = train_data)

# Enregistrement de l'accuracy

    A.append(a)
    index.append(j)

#Affichage de l'accuracy

plt.plot(index, A)
plt.show()
