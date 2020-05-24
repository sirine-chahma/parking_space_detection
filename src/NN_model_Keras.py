#Création de l'algorithme d'apprentissage automatique fully connected à partir de la bibliothèque Tensorflow et avec l'API Keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Chargement de la base de données pour cet algorithme

#Ouverture des images dans images
pickle_in = open("X.pickle", "rb")
images = pickle.load(pickle_in)

#Ouverture des labels dans targets_init
pickle_in = open("y.pickle", "rb")
targets_init = pickle.load(pickle_in)


#Modification des labels pour que une place busy ne soit plus 0 mais [1,0] et une place free ne soit plus 1 mais [0,1]
targets_bin = []

for i in range(len(targets_init)):
    if targets_init[i] == 0:
        targets_bin.append([1, 0])
    if targets_init[i] == 1:
        targets_bin.append([0, 1])


#Séparation de la base de données en 2 : 80% servent à l'entraînement de l'algortihme et 20% servent à la validation
images_train, images_test, targets_train, targets_test = train_test_split(images, targets_bin, test_size=0.2, random_state = 1)

#Conversion des différents objets en array (pour être sûr)
targets_test = np.array(targets_test)
targets_train = np.array(targets_train)
images_train = np.array(images_train)
images_test = np.array(images_test)

# Redimensionner la base de données et la convertir en flottant
images_train = images_train.reshape(-1, 150 * 150)
images_train = images_train.astype(float)
images_test = images_test.reshape(-1, 150 * 150)
images_test = images_test.astype(float)

#Normaliser les valeurs des pixels de chaque image pour qu'ils aient une moyenne et un écart-type plus faible pour que la convergence de l'algorithme soit plus rapide
scaler = StandardScaler()
images_train = scaler.fit_transform(images_train)
images_test = scaler.transform(images_test)

targets_names = ["busy", "free"]

# Afficher une image
plt.imshow(np.reshape(images_train[5], (150, 150)), cmap="binary")
plt.title(targets_names[targets_init[5]])
plt.show()

# Mise en place du model séquentiel
model = tf.keras.models.Sequential()

# Ajout des couches de neuronnes
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(2, activation="softmax"))


# Compilation du modèle

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'])


#Enregistrement des loss et des accuracy de notre algorithme pour l'entraînement et pour la validation pour chaque époque d'entraînement
History = model.fit(images_train, targets_train, epochs=2, validation_split=0.2)


loss_curve = History.history['loss']
acc_curve = History.history['accuracy']


loss_val_curve = History.history['val_loss']
acc_val_curve = History.history['val_accuracy']

#Affichage des courbes d'erreur et d'accuracy

plt.plot(loss_curve, label='Train')
plt.plot(loss_val_curve, label='Val')
plt.legend(loc='upper left')
plt.title('Loss')
plt.show()

plt.plot(acc_curve, label="Train")
plt.plot(acc_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.show()

loss, acc = model.evaluate(images_test, targets_test)
print("Test loss", loss)
print("Train accuracy", acc)

#Sauvegrde du modèle
model.save('simple_nn.h5')
