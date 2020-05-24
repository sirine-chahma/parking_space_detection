import cv2
ImageFile = 'C:\\Users\paulm\Desktop\photo.etude.jpg'
img = cv2.imread(ImageFile)
hauteur,longueur=img.shape[:2]
print(hauteur,longueur)
y=100
h=250
x=200
w=150
i=0
while x+w<longueur:
    crop_img = img[y:y+h, x:x+w]
    cv2.imwrite('C:\\Users\paulm\Desktop\Places\img{}.jpg'.format(str(i)), crop_img)
    cv2.waitKey(0)
    x+=w
    i=i+1