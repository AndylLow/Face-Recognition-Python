import cv2
import numpy as np 

# Initialize the static image object
img = cv2.imread('faces_dataset/test/image_0020.jpg')
cv2.imshow('Original', img)

# Define the image size scaling factor
scaling_factor = 0.5

# Loop until you hit the Esc key, this will keep the windows from closing
while True:
    # Resize the frame
    frame = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)

    # Display the image
    cv2.imshow('Resized', frame)

    # Detect if the Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break
        
# Close all active windows
cv2.destroyAllWindows()
