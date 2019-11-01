import multiprocessing as mp
from recognition import Recognition
from loguru import logger
from bsutils.board import BoardSerial


class App:

    def __init__(self):
        self.s = BoardSerial()
        self.pattern = '$PNEUL,C'

    def run(self):

        board = self.s.open_connection('/dev/ttyUSB0')

        if board is None:
            logger.error('Serial port error')

        receive, send = mp.Pipe()
        actions = mp.Array('i', [1, 1, 0, 0, 1])

        s_service = mp.Process(target=self.communication, args=(board, actions, send))

        r = Recognition()
        r_service = mp.Process(target=r.recognition,
                               args=(board, actions, receive, 0, True))

        s_service.start()
        r_service.start()

        s_service.join()
        r_service.join()

    def communication(self, serial, actions, data):

        logger.info('Serial start')

        while True:

            if serial is None:
                serial = self.s.open_connection(port='/dev/ttyUSB0')
            else:

                while actions[0]:

                    try:

                        if serial.inWaiting() > 0:

                            message = serial.readline().decode('utf-8', 'replace')

                            if message[:8] == self.pattern:

                                self.s.send_message(serial, self.s.SEND_OK)

                                idx = message.find('*')

                                if idx != -1:

                                    if self.s.digit_verify(message) is True:

                                        if message[9] == '0':

                                            actions[1] = 0
                                            actions[2] = 1

                                            data.send(message)
                                        elif message[9] == '1':

                                            actions[1] = 0
                                            actions[3] = 1

                                        else:
                                            pass

                    except UnicodeError:
                        logger.error('Unicode error')
                    except IOError:
                        logger.error('Cable disconnected')


if __name__ == '__main__':
    a = App()
    a.run()
