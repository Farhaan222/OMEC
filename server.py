from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import model_from_json
import time
app = Flask(__name__,template_folder='template')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/getting', methods=['POST'])
def getting():
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# Loading the model
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model/emotion_model.h5")
#Turning on the camera
    cap = cv2.VideoCapture(0)
    ne=0
    dis=0
    hap=0
    ang=0
    fe=0
    s=0
    sur=0
    timeout = 15   # [seconds]
    timeout_start = time.time()
#A 15 seconds loop for recording the session
    while time.time() < timeout_start + timeout:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #haarcascade for drawing squares around the face
        ret, frame = cap.read()
        if(type(frame) == type(None)):
            break
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # To detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        # Predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            if(emotion_dict[maxindex]=="Neutral"):
                ne+=1
            elif(emotion_dict[maxindex]=="Disgusted"):
                dis+=1
            elif(emotion_dict[maxindex]=="Angry"):
                ang+=1
            elif(emotion_dict[maxindex]=="Fearful"):
                fe+=1
            elif(emotion_dict[maxindex]=="Happy"):
                hap+=1
            elif(emotion_dict[maxindex]=="Sad"):
                s+=1
            elif(emotion_dict[maxindex]=="Surprised"):
                sur+=1
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    #Calculation part
    total=ne+fe+ang+sur+s+hap+dis
    if ne!=0:
        ne=round((ne/total)*100,2)
    if fe!=0:
        fe=round((fe/total)*100,2)
    if ang!=0:
        ang=round((ang/total)*100,2)
    if sur!=0:
        sur=round((sur/total)*100,2)
    if s!=0:
        s=round((s/total)*100,2)
    if hap!=0:
        hap=round((hap/total)*100,2)
    if dis!=0:
        dis=round((dis/total)*100,2)
    cap.release()
    cv2.destroyAllWindows()
    answer=""+str(ne)+" "+str(fe)+" "+str(ang)+" "+str(sur)+" "+str(s)+" "+str(hap)+" "+str(dis)+" "
    return render_template('output.html',a=ang,h=hap,d=dis,s=s,sur=sur,f=fe,n=ne)
app.run()