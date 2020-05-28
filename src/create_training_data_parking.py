#Création de la base de données pour notre cas d'étude

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pickle
import sys
# lissage niveau de gris
import ImageOps
import os
from PIL import Image
import cv2

DATADIR = "../data/parking"

CATEGORIES = ["busy", "free"]
IMG_SIZE = 150

training_data = []


def create_training_data():
    for category in CATEGORIES:  # parcourir "free" et "busy"

        path = os.path.join(DATADIR, category)  # crée un chemin vers "free" et "busy"
        class_num = CATEGORIES.index(category)  # classification entre libre et occupée  (0 ou 1). 0=busy 1=free

        for img in tqdm(os.listdir(path)):  # itérer sur chaque image pour les places libres et les places occupées

            # ouverture du fichier image
            ImageFile = 'D:/data/parking/{}/{}'.format(category, img)
            try:
                img = Image.open(ImageFile)
                imgF = Image.new(img.mode, img.size)

            except IOError:
                print('Erreur sur ouverture du fichier ' + ImageFile)

                sys.exit(1)
            # affichage des caractéristiques de l'image
            colonne, ligne = img.size
            format = img.format
            mode = img.mode

            # filtrage par convolution

            # passe haut
            Filtre = [[0,-4,0],[-4,18,-4],[0,-4,-0]]

            def Convolution2D(Filtre,TPix,x,y):
                p0 = p1 = p2 = 0
                for i in range(-1,1):
                    for j in range(-1,1):
                        p0 += Filtre[i+1][j+1]*TPix[y+i,x+j][0]
                        p1 += Filtre[i+1][j+1]*TPix[y+i,x+j][1]
                        p2 += Filtre[i+1][j+1]*TPix[y+i,x+j][2]
                p0 = int(p0 / 9.0)
                p1 = int(p1 / 9.0)
                p2 = int(p2 / 9.0)
                     # retourne le pixel convolué
                return (p0, p1, p2)

            TabPixel = img.load()
            for x in range(1,ligne-1):
                for y in range(1, colonne-1):
                    p = Convolution2D(Filtre, TabPixel, x, y)
                    imgF.putpixel((y, x), p)
            image = ImageOps.grayscale(imgF)
            try:
                # pour les images filtrées
                img = np.asarray(image, dtype='int32') #converti une image en array
                new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # redimensionner pour normaliser la base de données
                training_data.append([new_array, class_num])
            except Exception as e:  # in the interest in keeping the output clean...
                pass


create_training_data()

random.shuffle(training_data)  #mélange de la base de données


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Transformation des listes de 1 élément pour chaque pixel en flottant

X_f = []

for i in range(len(X)):
    X_f.append([])
    for k in range(len(X[i])):
        X_f[i].append([])
        for j in range(len(X[i][k])):
            X_f[i][k].append(X[i][k][j][0])

y = np.array(y)
X_f = np.array(X_f)

print("longueur de la base de données", len(X_f))

#afficher les images de la base de données

for i in range(len(X_f)):
    plt.imshow(X_f[i], cmap="binary")
    plt.show()

#Enregistrement des images dans X_parking.pickle

pickle_out = open("X_parking.pickle", "wb")
pickle.dump(X_f, pickle_out)
pickle_out.close()

#Enregistrement des labels dans Y_parking.pickle

pickle_out = open("y_parking.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

