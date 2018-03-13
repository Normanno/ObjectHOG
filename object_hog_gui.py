#!/usr/bin/env python

import Tkinter as tk
from Tkinter import *
import tkMessageBox
import tkFileDialog
from sklearn.externals import joblib
import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageTk
import threading
import multiprocessing as mp
import functools as ft
from image_processing.feature_extraction import feature_extraction
from image_processing.preprocessing import preprocess
import imutils
import cv2
from tkFileDialog import askopenfilename

from image_processing.divide_image import resize_image

class Application(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        #self.grid()
        self.model_file_types = [("Pickle files", "*.pkl")]
        self.image_file_types = [("Jpeg", "*.jpg"), ("PNG", "*.png")]
        self.window_width = 640
        self.window_height = 512
        self.actual_model_path = ''
        self.actual_classes_file = ''
        self.classes_dict = dict()
        self.classifier = None
        self.master.title('HOG GUI')
        #Sliding window parameters
        self.sliding_window_width = 64
        self.sliding_window_height = 128
        self.sliding_window_horizontal_step = 32
        self.sliding_window_vertical_step = 64
        self.actual_view_width = -1
        self.actual_view_height = -1
        self.top_correction = 0
        self.right_correction = 0
        self.bottom_correction = 0
        self.left_correction = 0
        #Widgets
        self.camera = None
        self.panel = None
        self.video_stream = None
        self.thread = None
        self.frame = None
        self.options_window = None
        self.sliding_window_options_frame = None
        self.sliding_window_height_entry = None
        self.sliding_window_width_entry = None
        self.sliding_window_horizontal_step_entry = None
        self.sliding_window_vertical_step_entry = None
        self.stopEvent = threading.Event()
        self.bottom_video_frame = None
        self.bottom_image_frame = None
        #init
        self.init_window()
        #self.calc_sliding_window_corrections()

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
        self.panel = Label(self.master)
        self.panel.grid(row=0, column=0, padx=30, pady=30)
        play_button = Button(self.bottom_video_frame, text="Play", command=self.start_video)
        play_button.grid(row=0, column=0)
        stop_button = Button(self.bottom_video_frame, text="Stop", command=self.stop_video)
        stop_button.grid(row=0, column=1)
        save_image = Button(self.bottom_video_frame, text="Save Iamge", command=self.save_video_frame)
        save_image.grid(row=0, column=2)
        choose_img_button = Button(self.bottom_image_frame, text="Choose Image", command=self.choose_image)
        choose_img_button.grid(row=0, column=0)
        classify_image = Button(self.bottom_image_frame, text="Classify", command=self.classify_sliding_window)
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
            self.panel = tk.Label(self.master)
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
        image = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        #image = preprocess(self.frame)
        #image = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
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
                self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
                self.frame = imutils.resize(self.frame, width=self.window_width, height=self.window_height)

                if self.actual_view_height < 0 or self.actual_view_width < 0:
                    self.actual_view_height, self.actual_view_width = self.frame.shape
                    self.calc_sliding_window_corrections()

                if self.classifier is not None:
                    image = self.classify_sliding_window(self.frame)
                else:
                    image = preprocess(self.frame)
                image = cv.flip(image, 1)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                self.frame = image
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
                key = int(fields[1])
                self.classes_dict[key] = fields[0]

    def calc_sliding_window_corrections(self):
        left_correction = (self.actual_view_width % self.sliding_window_horizontal_step) / 2
        right_correction = (self.actual_view_width% self.sliding_window_horizontal_step) / 2
        if ((self.actual_view_width % self.sliding_window_horizontal_step) % 2) == 1:
            left_correction = (self.actual_view_width % self.sliding_window_horizontal_step) / 2
            right_correction = ((self.actual_view_width % self.sliding_window_horizontal_step) / 2) + 1

        top_correction = (self.actual_view_height % self.sliding_window_vertical_step) / 2
        bottom_correction = (self.actual_view_height % self.sliding_window_vertical_step) / 2
        if ((self.actual_view_height % self.sliding_window_vertical_step) % 2) == 1:
            top_correction = (self.actual_view_height % self.sliding_window_vertical_step) / 2
            bottom_correction = ((self.actual_view_height % self.sliding_window_vertical_step) / 2) + 1

        self.top_correction = top_correction
        self.right_correction = right_correction
        self.bottom_correction = bottom_correction
        self.left_correction = left_correction

    def classify_sliding_window(self, image):
        top_y = lambda x: (x*self.sliding_window_horizontal_step) + self.right_correction
        bottom_y = lambda x: (top_y(x) + self.sliding_window_width)
        top_x = lambda x: (x * self.sliding_window_vertical_step) + self.top_correction
        bottom_x = lambda x: (top_x(x) + self.sliding_window_width)
        num_columns = ((self.actual_view_width - self.left_correction - self.right_correction - self.sliding_window_width)
                       / self.sliding_window_horizontal_step) + 1
        num_rows = ((self.actual_view_height - self.top_correction - self.bottom_correction - self.sliding_window_height)
                    / self.sliding_window_vertical_step) + 1
        result = np.ndarray(shape=(num_rows, num_columns), dtype=int)
        result.fill(-1)
        for i in range(0, num_rows):
            for j in range(0, num_columns):
                try:
                    self.classify(i, j, image, result)
                except ValueError:
                    print " Value error "+str(i)+" - "+str(j)
        image = cv.rectangle(image, (0, 0), (self.sliding_window_width, self.sliding_window_height), (0, 0, 255), 2)
        for i in range(0, num_rows):
            for j in range(0, num_columns):
                res = int(result[i][j])
                if res != -1 and str(self.classes_dict[res]).lower() != 'none':
                    image = cv.rectangle(image, (top_y(j), top_x(i)), (bottom_y(j), bottom_x(i)), (0, 0, 255), 2)
                    #todo smaller image text
                    image = cv.putText(image, self.classes_dict[res], (top_y(j)+2, top_x(i)+2),
                                             4, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        return image

    def classify(self, i, j, image, result_mat):
        left_x = (i * self.sliding_window_vertical_step) + self.top_correction
        right_x = left_x + self.sliding_window_height
        top_y = (j * self.sliding_window_horizontal_step) + self.left_correction
        bottom_y = top_y + self.sliding_window_width
        sub_img = image[left_x:right_x, top_y:bottom_y]

        #todo Allow bigger area to be resized to model dimensions

        feats = feature_extraction(sub_img)

        res = self.classifier.predict(feats.reshape([1, -1]))
        result_mat[i][j] = res

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
            self.calc_sliding_window_corrections()
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
