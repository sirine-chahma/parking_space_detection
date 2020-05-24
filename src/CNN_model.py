# réalisation d'un réseau de neuronnes convolutif

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle

#Chargement de la base de données pour cet algorithme

#Ouverture des images dans images

pickle_in = open("X_CNN.pickle", "rb")
images = pickle.load(pickle_in)

#Ouverture des labels dans targets_init

pickle_in = open("y_CNN.pickle", "rb")
targets_init = pickle.load(pickle_in)

#On ne prend qu'une partie des images car le traitement est trop long sinon
max_size = 200
images = images[:max_size]
targets_init = targets_init[:max_size]


# Evaluation de la taille de la base de données
size = images.shape
print('size', size)

#transforme les valeurs des images en flottants
images = images.astype(np.float32)

# séparation de la base de données en un lot d'entraînement et un lot de validation
images, images_valid, targets, targets_valid = train_test_split(images, targets_init, test_size=0.33)

draw_class = ["busy", "free"]

#Afficher une image (ici la 5ème) et la catégorie qui lui correspond (busy ou free)
plt.imshow(np.reshape(images_valid[5], (150, 150)), cmap="binary")
plt.title(draw_class[targets_valid[5]])
plt.show()

#Normaliser les valeurs des pixels de chaque image pour qu'ils aient une moyenne et un écart-type plus faible pour que la convergence de l'algorithme soit plus rapide

scaler = StandardScaler()
scaled_images = scaler.fit_transform(images.reshape(-1, 150*150))
scaled_images_valid = scaler.transform(images_valid.reshape(-1, 150*150))

# Redimensionner la base de données

scaled_images = scaled_images.reshape(-1, 150, 150, 1)
scaled_images_valid = scaled_images_valid.reshape(-1, 150, 150, 1)

#Création de la base de données qui servira pour l'entraînemet du réseau de neuronnes

train_dataset = tf.data.Dataset.from_tensor_slices((scaled_images, targets))

#Création de la base de données qui servira pour la validation de l'entraînemet du réseau de neuronnes

valid_dataset = tf.data.Dataset.from_tensor_slices((scaled_images_valid, targets_valid))

#Création du modèle convolutif

class ConvModel(tf.keras.Model):

    def __init__(self):
        super(ConvModel, self).__init__()
        # Convolutions
        self.conv1 = tf.keras.layers.Conv2D(32, 9, activation='relu', name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(64, 4, activation='relu', name="conv2")
        self.conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu', name="conv3")

        # Aplatir la convolution
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        # couches de neuronnes fully connected
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.out = tf.keras.layers.Dense(2, activation='softmax', name="output")

    #Mise bout à bout des différents layers
    def call(self, image):
        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        flatten = self.flatten(conv3)
        d1 = self.d1(flatten)
        output = self.out(d1)
        return output

#Compilation du modèle

model = ConvModel()
model.predict(scaled_images[0:1])

#Définition de la loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')

# Accuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(image, targets):
    with tf.GradientTape() as tape:
        # Faire une prédiction sur le lot entier
        predictions = model(image)
        # Récupérer l'erreur sur les prédictions
        loss = loss_object(targets, predictions)
    # Calculer le gradient qui respecte l'erreur voulue
    gradients = tape.gradient(loss, model.trainable_variables)
    # Changer les poids du modèle
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(targets, predictions)

@tf.function
def valid_step(image, targets):
    predictions = model(image)
    t_loss = loss_object(targets, predictions)
    valid_loss(t_loss)
    valid_accuracy(targets, predictions)


#Faire tourner le modèle sur plusieurs époques

epoch = 15
batch_size = 30
b = 0
loss_val_curve = []
acc_val_curve=[]

for epoch in range(epoch):
    # Entraînement
    for images_batch, targets_batch in train_dataset.batch(batch_size):
        train_step(images_batch, targets_batch)
        template = '\r Batch {}/{}, Loss: {}, Accuracy: {}'
        print(template.format(
            b, len(targets), train_loss.result(),
            train_accuracy.result()*100
        ), end="")
        b += batch_size
    # Validation
    for images_batch, targets_batch in valid_dataset.batch(batch_size):
        valid_step(images_batch, targets_batch)

    template = '\nEpoch {}, Valid Loss: {}, Valid Accuracy: {}'
    loss_val_curve.append(valid_loss.result())
    acc_val_curve.append(valid_accuracy.result()*100)
    print(template.format(
        epoch+1,
        valid_loss.result(),
        valid_accuracy.result()*100)
    )
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    train_accuracy.reset_states()
    train_loss.reset_states()

# Afficher l'erreur

plt.plot(loss_val_curve, label='Val ')
plt.legend(loc='upper left')
plt.title('Loss')
plt.show()

# Afficher l'accuracy

plt.plot(acc_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.show()