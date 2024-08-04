import cv2

# Location of the Haar Cascade XML file
cascade_loc = "face.xml"

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cascade_loc)

# Check if the classifier loaded properly
if face_cascade.empty():
    raise IOError("Unable to load the face cascade classifier xml file.")

# Load the image
img = 'pm.png'
image = cv2.imread(img)

# Check if the image is loaded properly
if image is None:
    raise IOError("Unable to load the image file.")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,  # Adjust this value if needed for better accuracy
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (5, 0,255), 2)
    cv2.putText(image, "Detected Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

# Display the output
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image (optional)
output_path = 'output_pm.png'
cv2.imwrite(output_path, image)
