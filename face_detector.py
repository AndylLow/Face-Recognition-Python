import cv2
import numpy as np

# Load the face cascade file
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')

# Check if the face cascade file has been loaded
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

# Initialize image object and set it to gray scale
img = cv2.imread('faces_dataset/test/image_0019.jpg')

# Define the scaling factor
scaling_factor = 0.5
# resize the image
img = cv2.resize(img, (0,0), fx=scaling_factor, fy=scaling_factor) 
# display the original
cv2.imshow('Original', img)

# Run the face detector on the grayscale image
face_rects = face_cascade.detectMultiScale(img, 1.3, 5)

# Draw rectangles on the image
for (x,y,w,h) in face_rects:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

# Display the new image
cv2.imshow('Face Detector', img)

# Wait for keypress, then destroy the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

