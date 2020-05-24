#Création de la base de données pour l'entraînement et la validation de l'algorithme d'apprentissage automatique convolutif


import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

DATADIR = "D:/data/final"

CATEGORIES = ["busy", "free"]
IMG_SIZE = 150

training_data = []


def create_training_data():
    for category in CATEGORIES:  # parcourir "free" et "busy"

        path = os.path.join(DATADIR,category)  # crée un chemin vers "free" et "busy"
        class_num = CATEGORIES.index(category)  # classification entre libre et occupée  (0 ou 1). 0=busy 1=free
        for img in tqdm(os.listdir(path)):  # itérer sur chaque image pour les places libres et les places occupées
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  #convertit une image en array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))# redimensionner pour normaliser la base de données
                training_data.append([new_array, class_num])  # ajout à notre base de données d'entraînement
            except Exception as e:
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))


create_training_data()

random.shuffle(training_data) #mélange de la base de données

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_f = []

y = np.array(y)
X_f = np.array(X)

#Enregistrement des images dans X_CNN.pickle

pickle_out = open("X_CNN.pickle", "wb")
pickle.dump(X_f, pickle_out)
pickle_out.close()

#Enregistrement des labels dans Y_CNN_.pickle

pickle_out = open("y_CNN.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
