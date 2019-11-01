from train import Person
import os


class Test:

    def __init__(self):
        self.p = Person()

    def run(self):
        self.p.train(show_image=False)


if __name__ == '__main__':
    t = Test()
    t.run()
