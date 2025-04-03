import pyttsx3  
import datetime 
import speech_recognition as sr
import wikipedia       
import smtplib
import webbrowser as wb
import os  
import pyautogui
import psutil 
import pyjokes  
import requests, json  
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import logging
import numpy as np
import wolframalpha
#from yaml import scan
import imutils
from imutils.video import FPS, VideoStream


engine = pyttsx3.init()
engine.setProperty('rate', 190)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('volume', 1)


#change voice
def voice_change(v):
    x = int(v)
    engine.setProperty('voice', voices[x].id)
    speak("done sir")


#speak function
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


#time function
def time():
    Time = datetime.datetime.now().strftime("%H:%M:%S")
    speak("The current time is")
    speak(Time)


#date function
def date():
    year = int(datetime.datetime.now().year)
    month = int(datetime.datetime.now().month)
    date = int(datetime.datetime.now().day)
    speak("The current date is")
    speak(date)
    speak(month)
    speak(year)



def checktime(tt):
    hour = datetime.datetime.now().hour
    if ("morning" in tt):
        if (hour >= 6 and hour < 12):
            speak("Good morning sir")
        else:
            if (hour >= 12 and hour < 18):
                speak("it's Good afternoon sir")
            elif (hour >= 18 and hour < 24):
                speak("it's Good Evening sir")
            else:
                speak("it's Goodnight sir")
    elif ("afternoon" in tt):
        if (hour >= 12 and hour < 18):
            speak("it's Good afternoon sir")
        else:
            if (hour >= 6 and hour < 12):
                speak("Good morning sir")
            elif (hour >= 18 and hour < 24):
                speak("it's Good Evening sir")
            else:
                speak("it's Goodnight sir")
    else:
        speak("it's night sir!")


#welcome function
def wishme():
    speak("Welcome Back")
    hour = datetime.datetime.now().hour
    if (hour >= 6 and hour < 12):
        speak("Good Morning sir!")
    elif (hour >= 12 and hour < 18):
        speak("Good afternoon sir")
    elif (hour >= 18 and hour < 24):
        speak("Good Evening sir")
    else:
        speak("Goodnight sir")

    speak("Nanu at your service, Please tell me how can i help you?")



def wishme_end():
    speak("turning off")
    hour = datetime.datetime.now().hour
    if (hour >= 6 and hour < 12):
        speak("Good Morning")
    elif (hour >= 12 and hour < 18):
        speak("Good afternoon")
    elif (hour >= 18 and hour < 24):
        speak("Good Evening")
    else:
        speak("Goodnight.. Sweet dreams")
    quit()


#command by user function
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listing...")
        speak("Listing.....")
        r.pause_threshold = 0.5
        audio = r.listen(source)

    try:
        print("Recognizing...")
        speak("Recognizing....")
        query = r.recognize_google(audio, language='en-in')
        #speak(query)
        #print(query)
    except Exception as e:
        print(e)
        speak("Say that again please...")

        return "None"

    return query


#joke function
def jokes():
    j = pyjokes.get_joke()
    print(j)
    speak(j)



def personal():
    speak(
        "I am Nanu, version 1.0, I am our assistent, I am developed by Shashank rajput"
    )
    speak("Now i hope you know me")


###### object detection






def scan():
    query = takeCommand().lower()
    ASSETS_PATH = 'assets/'
    MODEL_PATH = os.path.join(ASSETS_PATH, 'frozen_inference_graph.pb')
    CONFIG_PATH = os.path.join(ASSETS_PATH, 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
    LABELS_PATH = os.path.join(ASSETS_PATH, 'labels.txt')
    SCORE_THRESHOLD = 0.4
    NETWORK_INPUT_SIZE = (300, 300)
    NETWORK_SCALE_FACTOR = 1

    
    
    logger = logging.getLogger('detector')
    logging.basicConfig(level=logging.INFO)

    # Reading coco labels
    with open(LABELS_PATH, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    logger.info(f'Available labels: \n{labels}\n')
    COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

# Loading model from file
    logger.info('Loading model from tensorflow...')
    ssd_net = cv2.dnn.readNetFromTensorflow(model=MODEL_PATH, config=CONFIG_PATH)
# Initiating camera
    logger.info('Starting video stream...')
    vs = VideoStream(src=0).start()
    #time.sleep(2.0)
    fps = FPS().start()

    while True:
        # Reading frames
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        height, width, channels = frame.shape

    # Converting frames to blobs using mean standardization
        blob = cv2.dnn.blobFromImage(image=frame,
                                 scalefactor=NETWORK_SCALE_FACTOR,
                                 size=NETWORK_INPUT_SIZE,
                                 mean=(127.5, 127.5, 127.5),
                                 crop=False)

    # Passing blob through neural network
        ssd_net.setInput(blob)
        network_output = ssd_net.forward()

    # Looping over detections
        for detection in network_output[0, 0]:
            score = float(detection[2])
            class_index = np.int(detection[1])
            label = f'{labels[class_index]}: {score:.2%}'
            

        # Drawing likely detections
            if score > SCORE_THRESHOLD:
                left = np.int(detection[3] * width)
                top = np.int(detection[4] * height)
                right = np.int(detection[5] * width)
                bottom = np.int(detection[6] * height)

                cv2.rectangle(img=frame,
                          rec=(left, top, right, bottom),
                          color=COLORS[class_index],
                          thickness=4,
                          lineType=cv2.LINE_AA)

                cv2.putText(img=frame,
                        text=label,
                        org=(left, np.int(top*0.9)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=COLORS[class_index],
                        thickness=2,
                        lineType=cv2.LINE_AA)
                speak(label)
        cv2.imshow("Detector", frame)

    #exit loop 
        if "stop scanning" in query or "stop detection" in query :
            sleep()
            break

    # Exit loop by pressing "q"
        elif cv2.waitKey(1) & 0xFF == ord("q"):
            break
           
        fps.update()

    fps.stop()
    logger.info(f'\nElapsed time: {fps.elapsed() :.2f}')
    logger.info(f' Approx. FPS: {fps.fps():.2f}')
    cv2.destroyAllWindows()
    vs.stop()

####### to capture image 



def click_photo():
    cap=cv2.VideoCapture(0)
    count = 0
    while True:
        ret,test_img=cap.read()
        if not ret :
            continue
        cv2.imwrite("frame%d.jpg" % count, test_img)     # save frame as JPG file
        #count += 1
        #resized_img = cv2.resize(test_img, (1000, 700))
        #cv2.imshow('face detection Tutorial ',resized_img)
        if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
            break
    cap.release()
    cv2.destroyAllWindows




def notes():
    note = takeCommand()
    file = open('notes.txt', 'w')
    speak("Sir, Should i include date and time")
    snfm = takeCommand()
    if 'yes' in snfm or 'sure' in snfm:
            strTime = datetime.datetime.now().strftime("% H:% M:% S")
            file.write(strTime)
            file.write(" :- ")
            file.write(note)
    else:
        file.write(notes)

'''
def sleeping():
    speak("for how much time you want to stop me from listening commands")
    a = int(takeCommand())
    time.sleep(a)
    speak(a)
'''


def sleep():
    query = takeCommand().lower()
    speak("would you like to stop sir say yes....")
    #a = int(takeCommand())
    if "yes stop" in query:
        
        exit()
        
    #time.sleep(a)
    #speak(a)
        
    elif "no" in query or "do not stop" in query: 
        speak("would you like to sleep for some time")
        a = int(takeCommand())
        time.sleep(a)
        speak(a)    




def nanu():    
    if __name__ == "__main__":
        wishme()
    while (True):
        query = takeCommand().lower()

        #time

        if ('time' in query):
            time()
    #date

        elif ('date' in query):
            date()

    #personal info
        elif ("tell me about yourself" in query):
            personal()
        elif ("about you" in query):
            personal()
        elif ("who are you" in query):
            personal()
        elif ("yourself" in query):
            personal()

        elif ("developer" in query or "tell me about your developer" in query
              or "father" in query or "who develop you" in query
              or "developer" in query):
            res = open("about.txt", 'r')
            speak("here is the details: " + res.read())

    #searching on wikipedia

        elif ('wikipedia' in query or 'what' in query or 'who' in query
              or 'when' in query or 'where' in query):
            speak("searching...")
            query = query.replace("wikipedia", "")
            query = query.replace("search", "")
            query = query.replace("what", "")
            query = query.replace("when", "")
            query = query.replace("where", "")
            query = query.replace("who", "")
            query = query.replace("is", "")
            result = wikipedia.summary(query, sentences=2)
            print(query)
            print(result)
            speak(result)

        elif ("search on google" in query or "open website" in query):
            speak("What should i search or open?")
            chromepath = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
            search = takeCommand().lower()
            wb.get(chromepath).open_new_tab(search + '.com')

    #sysytem logout/ shut down etc

        elif ("logout" in query):
            os.system("shutdown -1")
        elif ("restart" in query):
            os.system("shutdown /r /t 1")
        elif ("shut down" in query):
            os.system("shutdown /r /t 1")

    #play songs

        elif ("play songs" in query):
            speak("Playing...")
            songs_dir = "C:\\Music"
            songs = os.listdir(songs_dir)
            os.startfile(os.path.join(songs_dir, songs[1]))
            quit()

    #reminder function

        elif ("create a reminder list" in query or "reminder" in query):
            speak("What is the reminder?")
            data = takeCommand()
            speak("You said to remember that" + data)
            reminder_file = open("data.txt", 'a')
            reminder_file.write('\n')
            reminder_file.write(data)
            reminder_file.close()

    #reading reminder list

        elif ("do you know anything" in query or "remember" in query):
            reminder_file = open("data.txt", 'r')
            speak("You said me to remember that: " + reminder_file.read())


    #jokes
        elif ("tell me a joke" in query or "joke" in query):
            jokes()

    #Nanu features
        elif ("tell me your powers" in query or "help" in query
              or "features" in query):
            features = ''' i can help to do lot many things like..
            i can tell you the current time and date,
            i can create the reminder list,
            i can tell you non funny jokes,
            i can open any website,
            i can search the thing on wikipedia,
            i can change my voice from male to female and vice-versa
            And yes one more thing, My boss is working on this system to add more features...,
            tell me what can i do for you??
            '''
            print(features)
            speak(features)

        elif ("hii" in query or "hello" in query or "goodmorning" in query
              or "goodafternoon" in query or "goodnight" in query
              or "morning" in query or "noon" in query or "night" in query):
            query = query.replace("Nanu", "")
            query = query.replace("hi", "")
            query = query.replace("hello", "")
            if ("morning" in query or "night" in query or "goodnight" in query
                    or "afternoon" in query or "noon" in query):
                checktime(query)
            else:
                speak("what can i do for you")

    #changing voice
        elif ("voice" in query):
            speak("for female say female and, for male say male")
            q = takeCommand()
            if ("female" in q):
                voice_change(1)
            elif ("male" in q):
                voice_change(0)
        elif ("male" in query or "female" in query):
            if ("female" in query):
                voice_change(1)
            elif ("male" in query):
                voice_change(0)

        
        elif 'notes' in query or 'notepad' in query or 'blog' in  query:
            speak("What should i write, sir")   
            notes()

        elif 'show note' in query or 'open notepad' in query or 'show my blog' in  query:
            speak("your save note is")   
            file = open("notes.txt", "r")
            speak(file.read())
            speak(file.read(6))


        
        elif 'scan' in query or 'detect' in query or "start scanning " in query:
            speak("scanning started")
            scan()

        
        elif 'fine' in query or "good" in query:
            speak("It's good to know that your fine")

        
        elif 'how are you' in query :
            speak("I am fine, Thank you")
            speak("How are you, Sir")

        elif "who i am" in query:
            speak("If you talk then definitely your human.")

        elif "why you came to world" in query: 
            speak("Thanks to shashank. further It's a secret")

        elif ' love' in query:
            speak("It is 7th sense that destroy all other senses")

        elif "who are you" in query or 'reason for you' in query:
            speak("I am your virtual Ai assistant created by shashank rajput")

  
        elif "will you be my gf" in query or "will you be my bf" in query:
            speak("I'm not sure about, may be you should give me some time")
    
        elif "how are you" in query:
            speak("I'm fine, glad you me that")
    
        elif "i love you" in query:
            speak("It's hard to understand")
    
        elif "what is your name" in query or "who are you" in query:
            speak("my name is nanu")

        elif "what's your name" in query or "What is your name" in query:
            speak("My friends call me by , Nanu ")


    #exit function

        elif ('i am done' in query or 'bye bye Nanu' in query
              or 'go offline nanu' in query or 'bye' in query
              or 'nothing' in query):
            wishme_end()
        elif "calculate" in query:
            app_id = "Wolframalpha api id"
            client = wolframalpha.Client(app_id)
            indx = query.lower().split().index('calculate')
            query = query.split()[indx + 1:]
            res = client.query(' '.join(query))
            answer = next(res.results).text
            print("The answer is " + answer)
            speak("The answer is " + answer)


