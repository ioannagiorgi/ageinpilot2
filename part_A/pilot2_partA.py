#! /usr/bin/env python
# -*- encoding: UTF-8 -*-
import qi
import argparse
import sys, os, time
import numpy as np
import random
import math, almath
import zmq, json, pickle
from naoqi import ALProxy, ALBroker
import vision_definitions
import speech_recognition as sr
from scipy.io.wavfile import write
import cv2

import requests, json
from six.moves import urllib

from pydub import AudioSegment
import base64
from real_time_speech import *
import threading

PATH = 'part_A'

r = sr.Recognizer()

GREETING = ['animations/Stand/Gestures/Hey_1', 'animations/Stand/Gestures/BodyTalk_11']
POINT_LEFT = ['animations/Stand/Gestures/YouKnowWhat_5', 'animations/Stand/Gestures/Explain_2', 'animations/Stand/Gestures/Explain_3']
POINT_RIGHT = ['animations/Stand/Gestures/Give_3', 'animations/Stand/Gestures/Explain_4']

with open(PATH + '/' + 'userID.txt') as f:
    userID = f.readlines()
    print(userID)

URL = "https://nlpit.na.icar.cnr.it/nlu/uk/pilot2?" # this is the main server, please check if it works before the experiment. If not, comment this line

# URL = ""    # this is the backup server, if the main server does not work, uncomment this line

class Animation:
    def __init__(self):
        self.animationProxy = ALProxy("ALAnimationPlayer", robotIP, 9559)
    
        self.greetings = GREETING
        self.point_at = {'left' : POINT_LEFT, 'right' : POINT_RIGHT}
    
    def greet(self, asnc=False):
        x = self.greetings.pop(random.randint(0, len(self.greetings)-1))
        if len(self.greetings) == 0:
            self.greetings = GREETING
        self.animationProxy.run(x, _async=asnc)
        print (x)
        return x
        
    def pointing(self, direction, asnc=False):
        if direction not in ['left', 'right']:
            print('Wrong argument:', direction)
            return
        
        x = self.point_at[direction].pop(random.randint(0, len(self.point_at[direction])-1))
        if len(self.point_at[direction]) == 0:
            D = direction   # for readibility
            if D == 'left':
                self.point_at[D] = POINT_LEFT
            elif D == 'right':
                self.point_at[D] = POINT_RIGHT
        
        self.animationProxy.run(x, _async=asnc)
        
def compute_angles(obj_info, target, locked=True):
    p = -(obj_info[target]['coordinates'][0] * 0.02)/20
    p2 = -(obj_info[target]['coordinates'][1] * 0.02)/20
    a = math.atan2(p, 0.65)
    a2 = 0.2 - math.atan2(p2, 0.6)
    return a, a2

def record_video(videoProxy, seconds=5, robotIP="129.12.41.129"):
    t0 = time.time()
    videoProxy.startRecording("/home/nao/videos", "clip_nao")
    time.sleep(seconds)
    videoInfo = videoProxy.videoProxy.stopRecording()
    
    cmd = 'sshpass -p nao scp nao@'+robotIP+':/home/nao/videos/clip_nao.avi ' + PATH + '/clip_nao.avi'
    os.system(cmd)
    
    t1 = time.time() 
    print ("video retrieved and reaction obtained in " + str(t1-t0) + " seconds.")
    
class Nao:
    def __init__(self, robotIP):
        self.ip = robotIP
        #self.port = port
        self.animate = Animation()
        
        # Get the services ALNavigation, ALMotion, AlMemory
        self.navigationProxy = ALProxy("ALNavigation", robotIP, 9559)
        self.motionProxy = ALProxy("ALMotion", robotIP, 9559)       
        self.postureProxy = ALProxy("ALRobotPosture", robotIP, 9559)
        self.memoryProxy = ALProxy("ALMemory", robotIP, 9559)
        self.peopleDetection = ALProxy("ALPeoplePerception", robotIP, 9559)
        self.peopleDetection.setTimeBeforePersonDisappears(10)
        self.trackerProxy = ALProxy("ALTracker", robotIP, 9559)
        #self.faceProxy = ALProxy("ALFaceTracker", robotIP, 9559)
        self.audioProxy = ALProxy("ALAudioDevice", robotIP, 9559)
        self.recorderProxy = ALProxy("ALAudioRecorder", robotIP, 9559)
        self.audioPlayer = ALProxy("ALAudioPlayer", robotIP, 9559)
        self.speechProxy = ALProxy("ALSpeechRecognition", robotIP, 9559)
        self.speechProxy.setLanguage("English")
        self.tts = ALProxy("ALTextToSpeech", robotIP, 9559)
        self.tts.setParameter("speed", 80)       
        self.animatedSpeech = ALProxy("ALAnimatedSpeech", robotIP, 9559)
        # self.animatedSpeech.setParameter("speed", 80)   
        self.camProxy = ALProxy("ALPhotoCapture", robotIP, 9559)
        self.camProxy.setResolution(2)
        self.camProxy.setPictureFormat("png")
        self.videoProxy = ALProxy("ALVideoRecorder", robotIP, 9559)
        if self.videoProxy.isRecording():
            videoInfo = self.videoProxy.stopRecording()
        self.videoProxy.setResolution(2)
        self.videoDevice = ALProxy('ALVideoDevice', robotIP, 9559)
        self.autonomousLife = ALProxy("ALAutonomousLife", robotIP, 9559)

        # self.speechProxy.subscribe("SoundDetected")
        # self.got_sound = False   
        
        self.subscriberZones = None
        self.people_detected = False
        self.usr_id=[]
        self.label = None
        
        self.recog_demo = False
        
    
    def greeting(self):
        self.animate.greet()
        
    def listen(self):
        data = []
        self.speechProxy.pause(True)
        vocabulary = ["yes", "yeap", "alright", "it is", "yes it is", "it's clear", "no", "no it is not", "no it isn't", "no it's not", "no, repeat"]
        self.speechProxy.setVocabulary(vocabulary, False)
        
        self.speechProxy.subscribe("Test_ASR")
        self.memoryProxy.subscribeToEvent("WordRecognized", robotIP, "onWordRecognized")

        self.speechProxy.pause(False)   
        time.sleep(3)
        self.speechProxy.unsubscribe("Test_ASR")
        data = self.memoryProxy.getData("WordRecognized")
        print(self.memoryProxy.getData("ALSpeechRecognition/Status"))
        print("data: %s" % data)
        return data

    def onWordRecognised(self, value):
        if value[1] > 0.30:
            if value[0] in ["yes", "yeap", "alright", "it is", "yes it is", "it's clear"]:
                self.tts.say('Fantastic! Let us begin \\pau=100\\ Ask me anything about the supplements in front of you \\pau=40\\ and I will explain')
            elif value[0] in ["no", "no it is not", "no it isn't", "no it's not", "no, repeat"]:
                self.tts.say('No problem \\pau=40\\ I will repeat for you. \\pau=40\\. You have two supplements in front of you. Show me the supplement and ask me what you need to know about them, for example \\pau=60\\, what is their name, \\pau=60\\ what are their benefits \\pau=60\\ why do you need them and \\pau=40\\ how much to take from each supplement')

    def onWordRecognised1(self, value):
        if value[1] > 0.30:
            if value[0] in ["yes", "yeap", "alright", "it is", "yes it is", "it's clear"]:
                return 1
            elif value[0] in ["no", "no it is not", "no it isn't", "no it's not", "continue"]:
                return 0
   
    def crouch(self, speed=0.5):
        self.postureProxy.goToPosture("Stand", speed)

    def autonomous_life_off(self):
        """
        Switch autonomous life off
        .. note:: After switching off, robot stays in resting posture. After \
        turning autonomous life default posture is invoked
        """
        self.autonomousLife.setState("disabled")
        self.crouch()
        print("[INFO]: Autonomous life is off")

    def take_picture(self, img_name='object'):
        time.sleep(0.5)
        self.camProxy.takePicture('/home/nao/recordings/cameras/', 'image')
        print ('sshpass -p nao scp nao@'+robotIP+':/home/nao/recordings/cameras/image.png ' + PATH + '/'+ img_name + '.png')
        cmd = 'sshpass -p nao scp nao@'+robotIP+':/home/nao/recordings/cameras/image.png ' + PATH + '/'+ img_name + '.png'
        os.system(cmd)
        return(img_name)
    
    def process_frame(self):
        context = zmq.Context()
        print ('frame processing: connecting.')
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")

        socket.send_json('ready')
        response = socket.recv_json()
        print(response)
        return(response)
    
    def close_yolo(self):
        context = zmq.Context()
        print ('frame processing: connecting.')
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")
        socket.send_json('end')
        done = socket.recv()
        print (done)
        
    def look_table(self):
        self.motionProxy.setStiffnesses("Head", 1.0)
        self.motionProxy.setStiffnesses("Body", 1.0)
        self.motionProxy.setAngles("HeadPitch", 0.3, 0.2)
        time.sleep(3.0)
        
    def look_at(self, yaw, pitch):
        self.motionProxy.setAngles("HeadYaw", yaw, 0.2)
        time.sleep(1)
        self.motionProxy.setAngles("HeadPitch", pitch, 0.2)
        time.sleep(1)
        
    def startMicrophonesRecording(self, filename, type, samplerate, channels):
        """ :param str filename: Name of the file where to record the sound.
            :param str type: wav or ogg.
            :param int samplerate: Required sample rate.
            :param AL::ALValue channels: vector of booleans.
        """
        print ('start recording...')
        self.recorderProxy.startMicrophonesRecording(filename, type, samplerate, channels)
        # time.sleep(5)

    def stopMicrophonesRecording(self):
        self.recorderProxy.stopMicrophonesRecording()  
        print ('record over')

    def playAudio(self, filename, param1, param2):
        self.audioPlayer.playFile(filename, param1, param2)             

    def start_behaviour(self):
        # self.animate.greet(True)
        
        time.sleep(0.5)
        self.tts.say('Welcome to the lab! \\pau=20\\')
        self.launchBehaviour('dialog_move_hands/animations/Wave01')
        self.stopBehaviour('dialog_move_hands/animations/Wave01')
        self.tts.say('I\'m NAO. \\pau=100\\ I\'ll assist you with your supplement intake today.')
        time.sleep(1)
        
        self.tts.say('On your left \\pau=10\\')
        self.launchBehaviour('dialog_move_hands/animations/OpenRHand')
        self.stopBehaviour('dialog_move_hands/animations/OpenRHand')
        self.tts.say('and on your right \\pau=10\\')
        self.launchBehaviour('dialog_move_hands/animations/OpenLHand')
        self.stopBehaviour('dialog_move_hands/animations/OpenLHand')        
        self.tts.say('there are two different supplements the dispenser gave you \\pau=100\\ I will explain to you what they are \\pau=20\\, how to use them \\pau=30\\ and when. You need to ask me about each supplement and I will describe it to you')
        self.tts.say("Is that clear?")
        user_speechdata = nao.listen()
        self.onWordRecognised(user_speechdata)
    
    def google_speech_recognition(self, filename):
        with sr.AudioFile(filename) as source:
            audio_text = r.listen(source)
            try:
            # using google speech recognition
                text = r.recognize_google(audio_text)
                print('Converting audio transcripts into text ...')
                print(text)
        
            except:
                text = 'Sorry.. run again...'
                print(text)

        return(text)
    
    def real_time_speech(self):
        # nao.stopMicrophonesRecording()
        tmp = None
        if Exception:
            self.stopMicrophonesRecording()

        record_path = '/home/nao/record.wav'
        timeout = 5
        print("[INFO]: Speech recognition is in progress. Say something.")
        print("[INFO]: Robot is listening to you")
        # self.tts.say("i am listening")

        while True:
            self.startMicrophonesRecording(record_path, 'wav', 16000, (1,0,0,0))
            time.sleep(2)
            self.take_picture()
            time.sleep(6)
            self.stopMicrophonesRecording()
            print ('sshpass -p nao scp nao@'+robotIP+':/home/nao/record.wav ' + PATH + '/record.wav')
            cmd = 'sshpass -p nao scp nao@'+robotIP+':/home/nao/record.wav ' + PATH + '/record.wav'
            os.system(cmd)

            speech_detected, average_intensity = VAD.moattar_homayounpour(PATH + '/record.wav', 0, 1)
            print (speech_detected, average_intensity)

            if speech_detected == True and average_intensity >= 25:
                userspeech_stt = self.google_speech_recognition(PATH + '/record.wav')
                print(userspeech_stt)
                if userspeech_stt == 'stop' or userspeech_stt == 'stop experiment':
                    self.tts.say("Are you sure you want to stop?")
                    user_speechdata = nao.listen()
                    if self.onWordRecognised1(user_speechdata):
                        self.tts.say("Glad to speak with you today. \\pau=50\\ Have a lovely day!")
                        #self.autonomousLife.setState("interactive")
                        self.postureProxy.goToPosture("Crouch", 0.5)
                        self.close_yolo()
                        self.stop_tracking()
                        quit()
                    else:
                        self.tts.say("What else would you like to know?")
                        continue; 
                if userspeech_stt == 'Sorry.. run again...':
                    self.tts.say("I am sorry \\pau=40\\ i did not understand you. \\pau=60\\ can you please repeat")
                else:
                    response =  self.process_frame()
                    print(response)
                    if response != 'False':
                        supp_index = response
                        tmp = supp_index
                        PARAMS = {'text':userspeech_stt, 'recognizedElement':supp_index, 'userID':userID}
                        r = requests.get(url = URL, params = PARAMS)
                        data = r.json()	
                        nao_replies = data["output"]["text"]
                        self.tts.say(str(nao_replies).encode('ascii', 'ignore'))  # speak the reply from NLP
                    elif response == 'False':
                        supp_index = tmp
                        print(supp_index)
                        PARAMS = {'text':userspeech_stt, 'recognizedElement':supp_index}
                        r = requests.get(url = URL, params = PARAMS)
                        data = r.json()	
                        nao_replies = data["output"]["text"]
                        print(nao_replies)
                        self.tts.say(str(nao_replies).encode('ascii', 'ignore'))  # speak the reply from NLP
            else:
                self.startMicrophonesRecording(record_path, 'wav', 16000, (1,0,0,0))
                time.sleep(6) #leave 6 seconds here, otherwise it will not wait enough before asking if the participant has more questions
                self.stopMicrophonesRecording()
                cmd = 'sshpass -p nao scp nao@'+robotIP+':/home/nao/record.wav ' + PATH  + '/record.wav'
                os.system(cmd)
                speech_detected, average_intensity = VAD.moattar_homayounpour(PATH + '/record.wav', 0, 1)
                print (speech_detected, average_intensity)
                if speech_detected == False or average_intensity < 30:
                    self.tts.say('Would you like to ask me something else? \\pau=30\\ If not, please say \\pau=20\\ Stop')

    def face_tracker(self, faceSize):
            self.motion = ALProxy("ALMotion", robotIP, 9559)
            self.tracker = ALProxy("ALTracker", robotIP, 9559)
                # First, wake up.
            self.motion.wakeUp()

            # Add target to track.
            targetName = "Face"
            faceWidth = faceSize
            self.tracker.registerTarget(targetName, faceWidth)

            # Then, start tracker.
            self.tracker.track(targetName)

            print ("ALTracker successfully started, now show your face to robot!")
            print ("Use Ctrl+c to stop this script.")


    def stop_tracking(self):
        # Stop tracker.
        self.tracker.stopTracker()
        self.tracker.unregisterAllTargets()
        self.motion.rest()

        print ("ALTracker stopped.")

    def launchBehaviour(self, behavior_name):
        self.behavior_mng_service = ALProxy("ALBehaviorManager", robotIP, 9559)
        names = self.behavior_mng_service.getInstalledBehaviors()
        # print ("Behaviors on the robot:")
        # print (names)

        # Launch and stop a behavior, if possible.
    # Check that the behavior exists.
        # Check that the behavior exists.
        if (self.behavior_mng_service.isBehaviorInstalled(behavior_name)):
            # Check that it is not already running.
            if (not self.behavior_mng_service.isBehaviorRunning(behavior_name)):
                # Launch behavior. This is a blocking call, use _async=True if you do not
                # want to wait for the behavior to finish.
                self.behavior_mng_service.runBehavior(behavior_name, _async=True)
                time.sleep(0.5)
            else:
                print ("Behavior is already running.")

        else:
            print ("Behavior not found.")
            return


        names = self.behavior_mng_service.getRunningBehaviors()
        print ("Running behaviors:")
        print (names)

    def stopBehaviour(self, behavior_name):
        # Stop the behavior.
        if (self.behavior_mng_service.isBehaviorRunning(behavior_name)):
            self.behavior_mng_service.stopBehavior(behavior_name)
            time.sleep(1.0)
        else:
            print ("Behavior is already stopped.")

        names = self.behavior_mng_service.getRunningBehaviors()
        print ("Running behaviors:")
        print (names)




if __name__ == '__main__':
    #robotIP='129.12.41.121'
    #port='9559'
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--robotIP", type=str, default='127.0.0.1',
                    #help="robot IP")

    #parser.add_argument("--facesize", type=float, default=0.1,
                    #help="Face width.")
    #args = parser.parse_args()
    #if len(sys.argv) <= 0:
    #    print ("Usage python alrobotposture.py robotIP (optional default: 127.0.0.1)")
    #else:
    #print(args.robotIP)

    #robotIP = args.robotIP

    #print("robotIP = ", robotIP)

    #nao = Nao(robotIP)

    #nao.autonomous_life_off()
    #nao.face_tracker(args.facesize)
    #nao.start_behaviour()
    #nao.real_time_speech()

    robotIP='192.168.1.102' #CHANGE THE IP HERE 192.168.1.102
    #robotIP='129.12.41.121'
    port='9559'
    parser = argparse.ArgumentParser()
    parser.add_argument("--facesize", type=float, default=0.1,
                    help="Face width.")
    args = parser.parse_args()
    if len(sys.argv) <= 1:
        print ("Usage python alrobotposture.py robotIP (optional default: 127.0.0.1)")
    else:
        robotIp = sys.argv[1]

    nao = Nao(robotIP)

    nao.autonomous_life_off()
    nao.face_tracker(args.facesize)
    #nao.start_behaviour()
    nao.real_time_speech()

    

    

