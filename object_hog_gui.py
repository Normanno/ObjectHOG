#!/usr/bin/env python

import Tkinter as tk
from Tkinter import *
import tkMessageBox
import tkFileDialog
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
        self.options_window = None
        self.sliding_window_options_frame = None
        self.sliding_window_width = 64
        self.sliding_window_height = 128
        self.sliding_window_horizontal_step = 32
        self.sliding_window_vertical_step = 64
        self.sliding_window_height_entry = None
        self.sliding_window_width_entry = None
        self.sliding_window_horizontal_step_entry = None
        self.sliding_window_vertical_step_entry = None
        self.stopEvent = threading.Event()
        self.bottom_video_frame = None
        self.bottom_image_frame = None
        self.init_window()
        self.create_widgets()

    def init_window(self):
        self.init_menu()
        #self.init_sliding_window()
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
        menu.add_cascade(label="Source", menu=source_menu)

        model_menu = Menu(menu)
        model_menu.add_command(label="Choose Model", command=self.choose_model_file)
        model_menu.add_command(label="Sliding window", command=self.display_sliding_window_options)
        menu.add_cascade(label="Classification", menu=model_menu)

    def init_sliding_window(self):
        self.sliding_window_options_frame = Frame(self.master)
        sliding_window_height_label = Label(self.sliding_window_options_frame, text="Sliding window height")
        sliding_window_height_label.grid(row=0, column=0)
        sliding_window_width_label = Label(self.sliding_window_options_frame, text="Sliding window width")
        sliding_window_width_label.grid(row=1, column=0)
        self.sliding_window_height_entry = Entry(self.sliding_window_options_frame)
        self.sliding_window_height_entry.grid(row=0, column=1)
        self.sliding_window_width_entry = Entry(self.sliding_window_options_frame)
        self.sliding_window_width_entry.grid(row=1, column=1)

        sliding_window_vertical_step_label = Label(self.sliding_window_options_frame, text="Sliding window vertical pass")
        sliding_window_vertical_step_label.grid(row=2, column=0)
        sliding_window_horizontal_step_label = Label(self.sliding_window_options_frame, text="Sliding window horizontal pass")
        sliding_window_horizontal_step_label.grid(row=3, column=0)
        self.sliding_window_vertical_step_entry = Entry(self.sliding_window_options_frame)
        self.sliding_window_vertical_step_entry.grid(row=2, column=1)
        self.sliding_window_horizontal_step_entry = Entry(self.sliding_window_options_frame)
        self.sliding_window_horizontal_step_entry.grid(row=3, column=1)

        save_button = Button(self.sliding_window_options_frame, text="Save", command=self.save_sliding_window_values)
        save_button.grid(row=4, column=0)
        save_and_exit_button = Button(self.sliding_window_options_frame, text="Save and Exit", command=self.save_sliding_window_values_and_exit)
        save_and_exit_button.grid(row=4, column=1)

    def create_widgets(self):
        print 'creating widgets'
        self.bottom_image_frame = Frame(self.master)
        self.bottom_video_frame = Frame(self.master)
        play_button = Button(self.bottom_video_frame, text="Play", command=self.start_video)
        play_button.grid(row=0, column=0)
        stop_button = Button(self.bottom_video_frame, text="Stop", command=self.stop_video)
        stop_button.grid(row=0, column=1)
        save_image = Button(self.bottom_video_frame, text="Save Iamge", command=self.save_video_frame)
        save_image.grid(row=0, column=2)
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

    def classify_single_object(self):
        print "select single object"

    #VIDEO
    def init_video_gui(self):
        print "init video gui"
        self.bottom_image_frame.grid_forget()
        self.bottom_video_frame.grid(row=5, column=0)
        self.start_video()

    def start_video(self):
        self.init_video_stream()
        self.stopEvent.clear()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()

    def stop_video(self):
        self.stopEvent.set()

    def save_video_frame(self):
        f = tkFileDialog.asksaveasfile(mode='w')
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        f.close()
        cv.imwrite(f.name, self.frame)

    def video_loop(self):
        try:
            while not self.stopEvent.is_set():
                ret, self.frame = self.video_stream.read()
                self.frame = imutils.resize(self.frame, width=self.window_width, height=self.window_height)
                if self.classifier is not None:
                    image = self.classify_all_image()
                else:
                    image = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
                    image = cv.flip(image, 1)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)
                self.update_panel(image)

        except RuntimeError, e:
            print("[INFO] caught a RuntimeError")
        if self.video_stream is not None:
            self.video_stream.release()
            self.video_stream = None

    #MODEL
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

    def classify_all_image(self):

        return self.frame

    def save_sliding_window_values(self):
        sw_h = 0
        sw_w = 0
        sw_vs = 0
        sw_hs = 0
        error = False

        try:
            sw_h = int(self.sliding_window_height_entry.get())
        except ValueError:
            tkMessageBox.showerror("Error", "Height must be an integer")
            error = True

        try:
            sw_w = int(self.sliding_window_width_entry.get())
        except ValueError:
            tkMessageBox.showerror("Error", "Width must be an integer")
            error = True

        try:
            sw_vs = int(self.sliding_window_vertical_step_entry.get())
        except ValueError:
            tkMessageBox.showerror("Error", "Vertical step must be an integer")
            error = True

        try:
            sw_h = int(self.sliding_window_horizontal_step_entry.get())
        except ValueError:
            tkMessageBox.showerror("Error", "Horizontal step must be an integer")
            error = True

        if not error:
            self.sliding_window_height = sw_h
            self.sliding_window_width = sw_w
            self.sliding_window_vertical_step = sw_vs
            self.sliding_window_horizontal_step = sw_hs
        return not error

    def save_sliding_window_values_and_exit(self):
        if self.save_sliding_window_values():
            self.options_window.destroy()

    def display_sliding_window_options(self):
        self.options_window = Toplevel(self.master)
        self.sliding_window_options_frame = Frame(self.options_window)
        self.sliding_window_options_frame.grid(row=0, column=0)
        sliding_window_height_label = Label(self.sliding_window_options_frame, text="Sliding window height")
        sliding_window_height_label.grid(row=0, column=0)
        sliding_window_width_label = Label(self.sliding_window_options_frame, text="Sliding window width")
        sliding_window_width_label.grid(row=1, column=0)
        self.sliding_window_height_entry = Entry(self.sliding_window_options_frame)
        self.sliding_window_height_entry.grid(row=0, column=1)
        self.sliding_window_height_entry.insert(0, self.sliding_window_height)
        self.sliding_window_width_entry = Entry(self.sliding_window_options_frame)
        self.sliding_window_width_entry.grid(row=1, column=1)
        self.sliding_window_width_entry.insert(0, self.sliding_window_width)

        sliding_window_vertical_step_label = Label(self.sliding_window_options_frame,
                                                   text="Sliding window vertical step")
        sliding_window_vertical_step_label.grid(row=2, column=0)
        sliding_window_horizontal_step_label = Label(self.sliding_window_options_frame,
                                                     text="Sliding window horizontal step")
        sliding_window_horizontal_step_label.grid(row=3, column=0)
        self.sliding_window_vertical_step_entry = Entry(self.sliding_window_options_frame)
        self.sliding_window_vertical_step_entry.grid(row=2, column=1)
        self.sliding_window_vertical_step_entry.insert(0, self.sliding_window_vertical_step)
        self.sliding_window_horizontal_step_entry = Entry(self.sliding_window_options_frame)
        self.sliding_window_horizontal_step_entry.grid(row=3, column=1)
        self.sliding_window_horizontal_step_entry.insert(0, self.sliding_window_horizontal_step)

        save_button = Button(self.sliding_window_options_frame, text="Save", command=self.save_sliding_window_values)
        save_button.grid(row=4, column=0)
        save_and_exit_button = Button(self.sliding_window_options_frame, text="Save and Exit",
                                      command=self.save_sliding_window_values_and_exit)
        save_and_exit_button.grid(row=4, column=1)

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
