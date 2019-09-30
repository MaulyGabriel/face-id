import os
import cv2
import glob
import imutils
import logging
import numpy as np
from time import time
import face_recognition as fr

logging.basicConfig(filename='train.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class Person:

    def __init__(self):

        self.codes = []
        self.faces = []
        self.names = []
        self.color = (0, 0, 0)
        self.name = ''
        self.status = ''
        self.dataset = 'images'
        self.models = {
            'codes': 'models/codes.npy',
            'images': 'models/images.npy',
            'names': 'models/names.npy'
        }

        self.log = '{} train: {} s'

    def train(self, show_image=False, resize=240):

        init = time()

        for file in glob.glob(os.path.join(self.dataset, '*jpg')):

            image = fr.load_image_file(file)
            image = imutils.resize(image, width=resize)

            wrote = fr.face_encodings(image)

            if len(wrote) > 0:

                self.faces.append(fr.face_encodings(image)[0])

                data = file.split('/')
                data = data[1].split('.')

                self.codes.append(data[0])
                self.names.append(data[1])
                self.status = 'ok'
                self.color = (0, 255, 0)
                self.name = data[1]
            else:

                self.status = 'error'
                self.name = file[7:]
                self.color = (0, 0, 255)

                logging.error(file[7:] + ': image error')

            if show_image:
                cv2.putText(image, self.status, (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)
                cv2.imshow(self.name, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        np.save(self.models['codes'], self.codes)
        np.save(self.models['images'], self.faces)
        np.save(self.models['names'], self.names)

        logging.error(self.log.format('Finish', round(time()-init, 2)))
