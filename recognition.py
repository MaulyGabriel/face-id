import os
import cv2
import dlib
import shutil
import imutils
import numpy as np
import face_recognition as fr
from time import sleep, time, strftime
from train import Person
from imutils.video import FPS
from imutils.video import VideoStream
from bsutils.board import BoardSerial
from loguru import logger


class Recognition:

    def __init__(self):

        self.person = Person()
        self.detector = dlib.get_frontal_face_detector()
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.camera = ''
        self.name = ''
        self.path = '/home/pi/prod/rec-facial/images/'
        self.old = '/home/pi/prod/rec-facial/images/old/'
        self.models = {

            'rasp': [
                '/home/pi/prod/rec-facial/models/codes.npy',
                '/home/pi/prod/rec-facial/models/images.npy',
                '/home/pi/prod/rec-facial/models/names.npy'
            ],

            'pc': [
                './models/codes.npy',
                './models/images.npy',
                './models/names.npy'
            ]
        }

        self.color = (255, 255, 0)
        self.white = (255, 255, 255)
        self.resize = 480
        self.board = BoardSerial()
        self.timeout = 180
        self.timestamp = strftime('%Y%m%d%H%M')
        self.codes, self.images, self.names = [], [], []

    def find_face(self, mean, stop, camera, fps):

        frame = np.array([])

        face = ''

        count = 0

        while len(face) < 1:

            count += 1

            frame = camera.read()
            face, confidence, idx = self.detector.run(frame, 0, mean)

            if count == stop:
                frame = np.array([])
                logger.error('Time\'s up')
                break

            fps.update()

        if frame.any():
            logger.error('successful registration')
        else:
            logger.error('error registering')

        return frame

    def load_models(self):

        try:

            return np.load(self.models['pc'][0]), np.load(self.models['pc'][1]), np.load(self.models['pc'][2])

        except FileNotFoundError:

            logger.error('Models not found')

            return None, None, None

    def recognition(self, serial, actions, data, device, show_image=False):

        old_person = 0
        alt_person = 0

        not_identified = 0
        not_detected = 0

        try:

            if None in self.load_models():
                logger.error('Models not found')
                exit()
            else:
                self.codes, self.images, self.names = self.load_models()

        except ValueError:
            self.codes, self.images, self.names = self.load_models()

        self.camera = VideoStream(src=device).start()
        fps = FPS().start()

        sleep(0.9)

        logger.info('Camera start')

        while True:

            if actions[3] and actions[4]:

                # change turn

                self.board.send_message(serial, self.board.SEND_OK)

                alt_person = old_person
                old_person = 0
                sleep(self.timeout)

                actions[4] = 1
                actions[3] = 0
                actions[4] = 0
                actions[1] = 1

            if actions[2]:

                # create a new person

                info = data.recv()
                person = self.find_face(mean=1.5, stop=30, camera=self.camera, fps=fps)

                if person.any():

                    self.codes = list(np.load(self.models['codes']))
                    self.names = list(np.load(self.models['names']))

                    info = info.spit(',')

                    code = info[3]

                    face = imutils.resize(person, width=480)

                    if code in self.codes:

                        code = self.codes.index(code)
                        name = self.names[code]

                        path_image = '{}{}.{}.jpg'.format(self.path, code, name)
                        path_move = '{}{}.{}.{}.jpg'.format(self.old, code, name, self.timestamp)

                        if os.path.isfile(path_image):
                            shutil.move(path_image, path_move)

                    final_name = '{}{}.{}.jpg'.format(self.path, code, info[4])

                    cv2.imwrite(final_name, face)

                    hexadecimal = '$PNEUD,C,0,{},{}'.format(code, info[4])

                    self.board.send_message(serial, hexadecimal)

                    self.person.train(show_image=None)

                    self.codes, self.images, self.names = self.load_models()

                else:
                    self.board.send_message(serial, 'PNEUD,C,0,-1')

                actions[1] = 1
                actions[2] = 0

            while actions[1]:

                # recognition a person

                init = time()

                codes_persons = []
                names_persons = []

                frame = self.camera.read()
                frame = imutils.resize(frame, width=480)
                frame_process = imutils.resize(frame, width=240)

                frame_process = frame_process[:, :, ::-1]

                face_locations = fr.face_locations(frame_process)
                face_encodings = fr.face_encodings(frame_process, face_locations)

                if len(face_locations) != 0:

                    for face_encoding in face_encodings:

                        matches = fr.compare_faces(self.images, face_encoding)
                        self.name = "Not identified"

                        if True in matches:
                            person_index = matches.index(True)
                            self.name = self.names[person_index]

                            codes_persons.append(self.codes[person_index])
                            names_persons.append(self.name)

                    codes_persons = list(set(codes_persons))
                    names_persons = list(set(names_persons))

                    if old_person == 0 and actions[3]:

                        if old_person in codes_persons:
                            idx = codes_persons.index(alt_person)
                            del codes_persons[idx]
                            del names_persons[idx]

                    if old_person in codes_persons:
                        not_detected = 0
                        not_identified = 0
                    else:

                        if len(codes_persons) > 0:

                            not_detected = 0
                            not_identified = 0

                            hexadecimal = '$PNEUD,C,1,{},{}'.format(str(codes_persons[0]), str(names_persons[0]))

                            self.board.send_message(serial, hexadecimal)

                            old_person = codes_persons[0]

                        else:
                            not_detected = 0

                            if not_identified == 20:
                                not_identified = 0
                            else:
                                not_identified += 1
                else:
                    not_identified = 0

                    if not_detected == 20:
                        not_detected = 0
                    else:
                        not_detected += 1

                if show_image:
                    cv2.imshow('Solinftec', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                fps.update()
