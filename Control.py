import cv2
import Module

img_name = "Images/True/two_people.jpg"
img = cv2.imread(img_name)

result1, result2 = Module.find_faces(img)

print("Frontal Faces found (x, y, w, h):")
print(result1)
print("Profile Faces found (x, y, w, h):")
print(result2)

for (x, y, w, h) in result1:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

for (x, y, w, h) in result2:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)


cv2.imshow('Result', img)


cv2.waitKey(0)
