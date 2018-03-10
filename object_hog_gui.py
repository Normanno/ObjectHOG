#!/usr/bin/env python

import Tkinter as tk
from Tkinter import *
import threading
from sklearn.externals import joblib
import cv2 as cv
from PIL import Image
from PIL import ImageTk
import Tkinter as tki
import threading
import datetime
import imutils
import cv2
from tkFileDialog import askopenfilename


class Application(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        #self.grid()
        self.model_file_types = [("Pickle files", "*.pkl")]
        self.image_file_types = [("Jpeg", "*.jpg"), ("PNG", "*.png")]
        self.window_width = 640
        self.window_height = 400
        self.actual_model_path = ''
        self.actual_classes_file = ''
        self.classes_dict = dict()
        self.classifier = None
        self.master.title('HOG GUI')
        self.camera = None
        self.panel = None
        self.video_stream = None
        self.thread = None
        self.frame = None
        self.stopEvent = threading.Event()
        self.bottom_video_frame = None
        self.bottom_image_frame = None
        self.init_window()
        self.create_widgets()

    def init_window(self):
        self.init_menu()
        self.create_widgets()

    def init_menu(self):
        menu = Menu(self.master)
        self.master.config(menu=menu)
        file_menu = Menu(menu)
        file_menu.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file_menu)

        source_menu = Menu(menu)
        source_menu.add_command(label="Camera", command=self.init_video_gui)
        source_menu.add_command(label="Image", command=self.init_image_gui)
#        source_menu.add_command(label="Start Camera", command=self.start_video)
#        source_menu.add_command(label="Stop Camera", command=self.stop_video)
#        source_menu.add_command(label="Classify all image", command=self.classify_all_image)
#        source_menu.add_command(label="Classify signle object", command=self.classify_single_object)
        menu.add_cascade(label="Source", menu=source_menu)

        model_menu = Menu(menu)
        model_menu.add_command(label="Choose model", command=self.choose_model_file)
        menu.add_cascade(label="Model", menu=model_menu)

    def create_widgets(self):
        print 'creating widgets'
        self.bottom_image_frame = Frame(self.master)
        self.bottom_video_frame = Frame(self.master)
        play_button = Button(self.bottom_video_frame, text="Play", command=self.start_video)
        play_button.grid(row=0, column=0)
        stop_button = Button(self.bottom_video_frame, text="Stop", command=self.stop_video)
        stop_button.grid(row=0, column=1)

        choose_img_button = Button(self.bottom_image_frame, text="Choose Image", command=self.choose_image)
        choose_img_button.grid(row=0, column=0)
        classify_image = Button(self.bottom_image_frame, text="Classify", command=self.classify_all_image)
        classify_image.grid(row=0, column=1)
        classify_object_button = Button(self.bottom_image_frame, text="Classify Object", command=self.classify_single_object)
        classify_object_button.grid(row=0, column=2)

    def init_video_stream(self, camera=0):
        print "init video stream "
        self.camera = camera
        if self.video_stream is not None:
            print "not none"
            self.stop_video()
        self.video_stream = cv2.VideoCapture(self.camera)

    def update_panel(self, image):
        if self.panel is None:
            self.panel = tki.Label(image=image)
            self.panel.image = image
            self.panel.grid(row=0, column=0, padx=30, pady=30)
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    #IMAGE
    def init_image_gui(self):
        print "init_image_gui"
        self.stop_video()
        self.bottom_video_frame.grid_forget()
        self.bottom_image_frame.grid(row=5, column=0)

    def choose_image(self):
        print "choose image"
        self.frame = cv.imread(askopenfilename(filetypes=self.image_file_types))
        self.frame = imutils.resize(self.frame, width=self.window_width, height=self.window_height)
        image = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        # if the panel is not None, we need to initialize it
        self.update_panel(image)

    def classify_all_image(self):
        print "select image"

    def classify_single_object(self):
        print "select single object"

    #VIDEO
    def init_video_gui(self):
        print "init video gui"
        self.bottom_image_frame.grid_forget()
        self.bottom_video_frame.grid(row=5, column=0)

    def start_video(self):
        self.init_video_stream()
        self.stopEvent.clear()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()

    def stop_video(self):
        self.stopEvent.set()

    def video_loop(self):
        try:
            while not self.stopEvent.is_set():
                ret, self.frame = self.video_stream.read()
                self.frame = imutils.resize(self.frame, width=self.window_width, height=self.window_height)
                # OpenCV represents images in BGR order
                # PIL represents images in RGB order
                #  then convert to PIL and ImageTk format
                image = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                self.update_panel(image)

        except RuntimeError, e:
            print("[INFO] caught a RuntimeError")
        if self.video_stream is not None:
            self.video_stream.release()
            self.video_stream = None

    #model
    def choose_model_file(self):
        self.actual_model_path = askopenfilename(title="Select model file", filetypes=self.model_file_types)
        print "Chosen model path : "+self.actual_model_path
        self.actual_classes_file = askopenfilename(title="Select classes labels file", filetypes=[("TXT", "*.txt")])
        print "Chosen classes path : "+self.actual_classes_file
        self.classifier = joblib.load(self.actual_model_path)
        self.classes_dict.clear()
        with open(self.actual_classes_file, "r") as cf:
            for line in cf:
                fields = line.replace("\n", "").split("\t")
                self.classes_dict[fields[0]] = fields[1]

    def client_exit(self):
        print "Exit from HOG GUI"
        self.stop_video()
        exit()


if __name__ == '__main__':
    print "starting object hog gui"
    root = Tk()
    root.geometry("700x500")
    app = Application(root)
    app.mainloop()
