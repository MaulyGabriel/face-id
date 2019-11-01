import os
import cv2
import glob
import imutils
from loguru import logger
import numpy as np
from time import time
import face_recognition as fr


class Person:

    def __init__(self):

        self.codes = []
        self.faces = []
        self.names = []
        self.color = (0, 0, 0)
        self.name = ''
        self.status = ''
        self.dataset = {
            'rasp': '/home/pi/prod/rec-facial/images',
            'pc': './images'
        }
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

        self.log = '{} train: {} s'

    def train(self, show_image=False, resize=240):

        init = time()

        for file in glob.glob(os.path.join(self.dataset['pc'], '*jpg')):

            image = fr.load_image_file(file)
            image = imutils.resize(image, width=resize)

            wrote = fr.face_encodings(image)

            if len(wrote) > 0:

                self.faces.append(fr.face_encodings(image)[0])

                data = file.split('/')[-1]
                data = data.split('.')
                logger.info(data)

                self.codes.append(data[0])
                self.names.append(data[1])
                self.status = 'ok'
                self.color = (0, 255, 0)
                self.name = data[1]
            else:

                self.status = 'error'
                self.name = file[7:]
                self.color = (0, 0, 255)

                logger.error(file[7:] + ': image error')

            if show_image:
                cv2.putText(image, self.status, (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)
                cv2.imshow(self.name, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        np.save(self.models['pc'][0], self.codes)
        np.save(self.models['pc'][1], self.faces)
        np.save(self.models['pc'][2], self.names)

        logger.success(self.log.format('Finish', round(time() - init, 2)))

    def train_rasp(self, show_image=False, resize=240):

        init = time()

        for file in glob.glob(os.path.join(self.dataset['rasp'], '*jpg')):

            image = fr.load_image_file(file)
            image = imutils.resize(image, width=resize)

            wrote = fr.face_encodings(image)

            if len(wrote) > 0:

                self.faces.append(fr.face_encodings(image)[0])

                data = file.split('/')
                data = data[6].split('.')

                self.codes.append(data[0])
                self.names.append(data[1])
                self.status = 'ok'
                self.color = (0, 255, 0)
                self.name = data[1]
            else:

                self.status = 'error'
                self.name = file[7:]
                self.color = (0, 0, 255)

                logger.error(file[7:] + ': image error')

            if show_image:
                cv2.putText(image, self.status, (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)
                cv2.imshow(self.name, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        np.save(self.models['rasp'][0], self.codes)
        np.save(self.models['rasp'][1], self.faces)
        np.save(self.models['rasp'][2], self.names)

        logger.success(self.log.format('Finish', round(time() - init, 2)))
