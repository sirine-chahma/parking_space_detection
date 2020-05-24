#Création de la base de données pour l'entraînement et la validation de l'algorithme d'apprentissage automatique fully connected + filtrage

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pickle
import sys
import ImageOps
import os
from PIL import Image
import cv2


DATADIR = "D:/data/final"

CATEGORIES = ["busy", "free"]
IMG_SIZE = 150

training_data = []


def create_training_data():
    for category in CATEGORIES:  # parcourir "free" et "busy"

        path = os.path.join(DATADIR,category)  # crée un chemin vers "free" et "busy"
        class_num = CATEGORIES.index(category)  # classification entre libre et occupée  (0 ou 1). 0=busy 1=free
        for img in tqdm(os.listdir(path)):  # itérer sur chaque image pour les places libres et les places occupées
            # ouverture du fichier image
            ImageFile = 'D:/data/final/{}/{}'.format(category, img)
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

#Application des filtres sur les images de la base de données

            # Image niveau de gris
            imgF=ImageOps.grayscale(img)

            # filtrage par convolution

            # 1)passe bas
            #Filtre = [[1, 1, 1], [1, 6, 1], [1, 1, 1]]

            # 2)passe haut
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
            for x in range(1, ligne-1):
                for y in range(1, colonne-1):
                    p = Convolution2D(Filtre, TabPixel, x, y)
                    imgF.putpixel((y, x), p)
            image = ImageOps.grayscale(imgF)

            # lissage des contours
            #imgF = image.filter(ImageFilter.CONTOUR)

            #img_array = cv2.imread(imgF, cv2.IMREAD_GRAYSCALE)  # convert to array
            #img = cv2.imread(ImageFile)
            #img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #ret, thresh1 = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)

            try:
                #pour les images filtrées
                image = np.asarray(image, dtype='int32') #converti une image en array
                new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) # redimensionner pour normaliser la base de données
                #pour les images non filtrées
                #img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array, class_num])  # ajout à notre training_data
            except Exception as e:  # in the interest in keeping the output clean...
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

#Transformation des listes de 1 élément pour chaque pixel en flottant

for i in range(len(X)):
    X_f.append([])
    for k in range(len(X[i])):
        X_f[i].append([])
        for j in range(len(X[i][k])):
            X_f[i][k].append(X[i][k][j][0])


y = np.array(y)
X_f = np.array(X_f)

#Affichage de la première image
for i in range(len(X_f)):
   plt.imshow(X_f[i], cmap="binary")
   plt.show()
   break

#Enregistrement des images dans X.pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X_f, pickle_out)
pickle_out.close()

#Enregistrement des labels dans Y.pickle
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()