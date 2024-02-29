from deepface import DeepFace
import cv2

## cannot manipulate the data
# DeepFace.stream(db_path='./dataset/allpics')

# Initialize the video capture object
cap = cv2.VideoCapture(1)

# Load the model and configure the backend
model_name = 'VGG-Face'
model = DeepFace.build_model(model_name)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    # Apply facial recognition
    try:
        # Analyze the frame (you can set enforce_detection to False if you're getting too many errors)
        result = DeepFace.find(frame, db_path='./dataset/yousuf', model_name=model_name, enforce_detection=False)

        print(result)
        # Loop through the found results
        for instance in result:
            identity = instance['identity']
            name = identity.split("/")[-2]  # Extract the name from the path, assuming identity is a file path

            # Draw a rectangle and put the name of the person
            face = instance['region']
            cv2.rectangle(frame, (face['x'], face['y']), (face['x']+face['w'], face['y']+face['h']), (255, 0, 0), 2)
            cv2.putText(frame, name, (face['x'], face['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    except Exception as e:
        print("Error in facial recognition:", e)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
