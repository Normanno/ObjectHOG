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
from image_processing.feature_extraction import feature_extraction
from image_processing.divide_image import resize_image
import imutils
from tkFileDialog import askopenfilename


class Application(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        #self.grid()
        self.model_file_types = [("Pickle files", "*.pkl")]
        self.image_file_types = [("Jpeg", "*.jpg"), ("PNG", "*.png")]
        self.window_width = 480
        self.window_height = 640
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
        self.panel = None
        self.original_image = None
        self.frame = None
        self.options_window = None
        self.sliding_window_options_frame = None
        self.sliding_window_height_entry = None
        self.sliding_window_width_entry = None
        self.sliding_window_horizontal_step_entry = None
        self.sliding_window_vertical_step_entry = None
        self.bottom_label_frame = None
        self.bottom_image_frame = None
        self.bottom_model_frame = None
        self.bottom_model_info = None
        self.options_panel = None
        self.model_center = IntVar()
        self.model_blurred = IntVar()
        self.classified_label = StringVar()
        self.selected_model = StringVar()
        #init
        self.init_window()
        #self.calc_sliding_window_corrections()

    def init_window(self):
        self.init_menu()
        self.init_sliding_window()
        self.create_widgets()
        self.panel.grid(row=0, column=0, padx=20, pady=10)
        self.bottom_label_frame.grid(row=6, column=0)
        self.bottom_model_frame.grid(row=7, column=0)
        self.bottom_image_frame.grid(row=8, column=0)
        self.bottom_model_info.grid(row=9, column=0)

    def init_menu(self):
        menu = Menu(self.master)
        self.master.config(menu=menu)
        #model_menu = Menu(menu)
        #model_menu.add_command(label="Load Model", command=self.choose_model_file)
        #model_menu.add_command(label="Sliding window", command=self.display_sliding_window_options)
        #menu.add_cascade(label="Classification", menu=model_menu)
        menu.add_command(label="Load Model", command=self.choose_model_file)
        menu.add_command(label="Exit", command=self.client_exit)

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
        self.bottom_model_frame = Frame(self.master)
        self.bottom_model_info = Frame(self.master)
        self.bottom_label_frame = Frame(self.master)
        self.panel = Label(self.master)

        label = StringVar()
        label.set("Classified as : ")
        label_classified = Label(self.bottom_label_frame, textvariable=self.classified_label, relief=RAISED)
        label_classified.grid(row=0, column=1)
        label_label = Label(self.bottom_label_frame, textvariable=label)
        label_label.grid(row=0, column=0)

        label_l = StringVar()
        label_l.set("Actual Model: ")
        label_model = Label(self.bottom_model_info, textvariable=self.selected_model, relief=FLAT)
        label_model.grid(row=0, column=1)
        label_label_model = Label(self.bottom_model_info, textvariable=label_l)
        label_label_model.grid(row=0, column=0)

        check_center = Checkbutton(self.bottom_model_frame, text="Model Center", variable=self.model_center, onvalue=1, offvalue=0, height=2, width=20)
        check_center.grid(row=0, column=0)
        check_blurred = Checkbutton(self.bottom_model_frame, text="Model Blur", variable=self.model_blurred,onvalue=1, offvalue=0, height=2, width=20)
        check_blurred.grid(row=0, column=1)

        choose_img_button = Button(self.bottom_image_frame, text="Choose Image", command=self.choose_image)
        choose_img_button.grid(row=0, column=0)
        #classify_image = Button(self.bottom_image_frame, text="Classify", command=self.classify_all_image)
        #classify_image.grid(row=0, column=1)
        classify_object_button = Button(self.bottom_image_frame, text="Classify Object", command=self.classify_single_object)
        classify_object_button.grid(row=0, column=2)

    def update_panel(self, image):
        if self.panel is None:
            self.panel = tk.Label(self.master)
            self.panel.image = image
            self.panel.grid(row=0, column=0, padx=20, pady=20)
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    #IMAGE
    def choose_image(self):
        file_path = askopenfilename(filetypes=self.image_file_types)
        print "file_path "+str(file_path)
        img = cv.imread(file_path)
        print "shape "+str(img.shape)
        if len(img.shape) > 2:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.original_image = img.copy()
        img_height, img_widht = img.shape
        if img_height > self.window_height:
            img = imutils.resize(img, height=self.window_height)
        #if img_widht > self.window_width:
        #    img = imutils.resize(img, width=self.window_width)
        print "img "+str(img.shape)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        # if the panel is not None, we need to initialize it
        self.frame = img
        self.update_panel(img)

    def classify_all_image(self):
        self.calc_sliding_window_corrections()
        if len(self.original_image.shape) > 2:
            self.original_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
        if self.actual_view_height < 0 or self.actual_view_width < 0:
            self.actual_view_height, self.actual_view_width = self.original_image.shape
            self.calc_sliding_window_corrections()
        image = imutils.resize(self.original_image, width=self.window_width, height=self.window_height)
        image = self.classify_sliding_window(self.original_image)
        self.original_image = image
        self.update_panel(self.original_image)

    def classify_single_object(self):
        centered = self.model_center.get() == 1
        blurred = self.model_blurred.get() == 1
        image = resize_image(self.original_image, centered=centered, blur_outer_borders=blurred)
        feats = feature_extraction(image)
        X_data = []
        X_label = self.classifier.predict(feats.reshape(1, -1))
        self.classified_label.set(self.classes_dict[int(X_label[0])])
        print "classified as " + self.classes_dict[int(X_label[0])]

    #MODEL
    def choose_model_file(self):
        self.actual_model_path = askopenfilename(title="Select model file", filetypes=self.model_file_types)
        print "Chosen model path : "+self.actual_model_path
        self.actual_classes_file = askopenfilename(title="Select classes labels file", filetypes=[("TXT", "*.txt")])
        print "Chosen classes path : "+self.actual_classes_file
        self.classifier = joblib.load(self.actual_model_path)
        self.classes_dict.clear()
        m_path = self.actual_model_path.rsplit("/", 1)[0]
        self.selected_model.set("..."+m_path[len(m_path)-30:])
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
        left_x = lambda x: (x * self.sliding_window_vertical_step) + self.top_correction
        right_x = lambda x: left_x(x) + self.sliding_window_height
        top_y = lambda y: (y * self.sliding_window_horizontal_step) + self.left_correction
        bottom_y = lambda y: top_y(y) + self.sliding_window_width

        num_columns = ((self.actual_view_width - self.left_correction - self.right_correction - self.sliding_window_width)
                       / self.sliding_window_horizontal_step) + 1
        num_rows = ((self.actual_view_height - self.top_correction - self.bottom_correction - self.sliding_window_height)
                    / self.sliding_window_vertical_step) + 1
        self.total_of_patches = num_rows * num_columns

        print "rows : "+str(num_rows)
        print "cols : "+str(num_columns)
        print "classification"
        X_data = []
        for i in range(0, num_rows):
            for j in range(0, num_columns):
                try:
                    sub_img = image[left_x(i):right_x(i), top_y(j):bottom_y(j)]
                    feats = feature_extraction(sub_img)
                    #process_pool.apply_async(ft.partial(classify, i, j, sub_img, self.classifier), callback=update_classification_results)
                    X_data.append(feats)

                except Exception, ValueError:
                    print " Value error "+str(i)+" - "+str(j)

        X_labels = self.classifier.predict(X_data)
        X_probs = self.classifier.predict_proba(X_data)

        image = self.draw_classification(image, X_probs, num_rows, num_columns)

        return image

    def draw_classification(self, image, results, num_rows, num_columns):
        top_y = lambda x: (x * self.sliding_window_horizontal_step) + self.right_correction
        bottom_y = lambda x: (top_y(x) + self.sliding_window_width)
        top_x = lambda x: (x * self.sliding_window_vertical_step) + self.top_correction
        bottom_x = lambda x: (top_x(x) + self.sliding_window_width)

        for i in range(0, num_rows):
            for j in range(0, num_columns):
                if len(results) > 0:
                    probs = results[i+j]
                    res = probs[probs.argmax(axis=0)]
                    if res > 0.5 and str(self.classes_dict[res]).lower() != 'none':
                        image = cv.rectangle(image, (top_y(j), top_x(i)), (bottom_y(j), bottom_x(i)), (0, 0, 255), 2)
                        image = cv.putText(image, self.classes_dict[res], (top_y(j)+2, top_x(i)+2),
                                                 4, 0.3, (255, 255, 255), 1, cv.LINE_AA)
        return image

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
    root.geometry("600x900")
    app = Application(root)
    app.mainloop()
