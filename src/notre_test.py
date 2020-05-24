#Test de notre algorithme sur notre cas d'étude

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

#Chargement du réseau de neuronnes

load_model = tf.keras.models.load_model('simple_nn.h5')

#Ouverture du fichier contenant les images à tester
pickle_in = open("X_parking.pickle", "rb")
images_parking = pickle.load(pickle_in)

#Ouverture du fichier contenant les labels des images testées (pour vérification)
pickle_in = open("y_parking.pickle", "rb")
targets_parking = pickle.load(pickle_in)

targets_names = ["busy", "free"]

# Redimensionner la base de données et la convertir en flottant

images_parking = images_parking.reshape(-1, 150 * 150)
images_parking = images_parking.astype(float)

#Normaliser les valeurs des pixels de chaque image pour qu'ils aient une moyenne et un écart-type plus faible pour que la convergence de l'algorithme soit plus rapide

scaler = StandardScaler()
images_parking_algo = scaler.fit_transform(images_parking)



for i in range(len(images_parking)):
    #Prédiction de l'occupation ou non de la place
    place = load_model.predict(images_parking_algo[i:i+1])

    if place[0][0] > place[0][1]: #Si la probabilité que la place soit occupée est plus grande que la probabilité que la place soit libre
        title =('prévu : '  + targets_names[0] + '   voulu  : ' + targets_names[targets_parking[i]]) #Prévu représente que cui est prévu par l'algorithme (ici occupé), et voulu représente ce qui est attendu
    else :
        title = ('prévu : ' + targets_names[1] + '    voulu : ' + targets_names[targets_parking[i]])
    # Affichage des images
    plt.imshow(np.reshape(images_parking[i], (150, 150)), cmap="binary")
    #Affichage des titres
    plt.title(title)
    plt.show()

