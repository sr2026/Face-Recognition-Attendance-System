import cv2
import numpy as np
from keras.models import load_model
from datetime import datetime
import csv

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model(r"C:/Users/KIIT/Desktop/face recognition/face recognition/final_model.h5")

labels = ['Adyajyoti', 'Ankit', 'Satyajit']  # Add your trained names
attendance = {}       # Dict: {name: {"entry": datetime, "exit": datetime}}

def get_pred_label(pred):
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255
    return img

def update_attendance(name):
    now = datetime.now()
    if name not in attendance:  
        # First appearance → Entry
        attendance[name] = {"entry": now, "exit": now}
    else:
        # Update last seen time → Exit
        attendance[name]["exit"] = now

def save_attendance():
    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for name, times in attendance.items():
            entry = times["entry"]
            exit_ = times["exit"]
            duration = (exit_ - entry).total_seconds() / 3600  # hours
            writer.writerow([name, entry.strftime('%Y-%m-%d'), 
                             entry.strftime('%H:%M:%S'), 
                             exit_.strftime('%H:%M:%S'),
                             f"{duration:.2f}"])
    print("Attendance saved to attendance.csv")

# ---- Use USB camera instead of IP camera ----
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = classifier.detectMultiScale(frame, 1.5, 5)
    
    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        
        pred = np.argmax(model.predict(preprocess(face)))
        name = get_pred_label(pred)
        
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        
        update_attendance(name)
        
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save attendance log
save_attendance()
