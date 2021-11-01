from tkinter import *
from tkinter import filedialog, NW
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class App:
    def __init__(self, parent, winTitle, winSize):
        self.window = parent
        self.window.title(winTitle)
        self.window.geometry(winSize)
        self.window.option_add('*tearOff', FALSE)

        self.containerLeft = None
        self.containerRight = None
        self.histogramFigure = None
        self.histogramCanvas = None

        self.menu = Menu(self.window)
        self.fileMenu = Menu(self.menu)
        self.editMenu = Menu(self.menu)
        self.primitiveMenu = Menu(self.editMenu)
        self.intensityMenu = Menu(self.editMenu)
        self.filterMenu = Menu(self.editMenu)
        self.redMenu = Menu(self.primitiveMenu)
        self.greenMenu = Menu(self.primitiveMenu)
        self.blueMenu = Menu(self.primitiveMenu)

        self.menu.add_cascade(label="File", menu=self.fileMenu)
        self.menu.add_cascade(label="Edit", menu=self.editMenu)

        self.fileMenu.add_cascade(label="Open", command=self.open_Image)
        self.fileMenu.add_cascade(label="Reset", command=self.resetEditedImg)
        self.fileMenu.add_cascade(label="Exit", command=self.window.destroy)

        self.editMenu.add_cascade(label="Primitive", menu=self.primitiveMenu)
        self.editMenu.add_cascade(label="Sampling", command=self.downSampling)
        self.editMenu.add_cascade(
            label="Quantization", command=self.quantization)
        self.editMenu.add_cascade(label="Intensity", menu=self.intensityMenu)
        self.editMenu.add_cascade(label="Negative", command=self.negativeImg)
        self.editMenu.add_cascade(
            label="Equalization", command=self.equalizeHistogram)
        self.editMenu.add_cascade(label="Filter", menu=self.filterMenu)

        self.primitiveMenu.add_cascade(label="Red", menu=self.redMenu)
        self.primitiveMenu.add_cascade(label="Green", menu=self.greenMenu)
        self.primitiveMenu.add_cascade(label="Blue", menu=self.blueMenu)

        self.intensityMenu.add_cascade(
            label="Intensity +", command=self.increase_brightness)
        self.intensityMenu.add_cascade(
            label="Intensity -", command=self.decrease_brightness)

        self.redMenu.add_cascade(
            label="Red +", command=lambda: self.addColorValue(0, 10))
        self.redMenu.add_cascade(
            label="Red -", command=lambda: self.subsColorValue(0, 10))

        self.greenMenu.add_cascade(
            label="Green +", command=lambda: self.addColorValue(1, 10))
        self.greenMenu.add_cascade(
            label="Green -", command=lambda: self.subsColorValue(1, 10))

        self.blueMenu.add_cascade(
            label="Blue +", command=lambda: self.addColorValue(2, 10))
        self.blueMenu.add_cascade(
            label="Blue -", command=lambda: self.subsColorValue(2, 10))

        self.filterMenu.add_cascade(
            label="LowPass", command=self.lowPassFilter)
        self.filterMenu.add_cascade(
            label="HighPass", command=self.highPassFilter)
        self.filterMenu.add_cascade(
            label="BandPass", command=self.bandPassFilter)

        self.window['menu'] = self.menu

        self.window.mainloop()

    def openFile(self):
        fileDir = filedialog.askopenfilename(title='Open an image', filetypes=[(
            'Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.svg *.gif')])
        return fileDir

    def open_Image(self,  size=[450, 450]):
        fileDir = self.openFile()
        self.currentFileDir = fileDir
        self.img = cv2.cvtColor(cv2.imread(fileDir), cv2.COLOR_BGR2RGB)
        self.editedImg = self.img
        self.photo = Image.fromarray(self.img)
        self.photo.thumbnail(size, Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image=self.photo)

        if(self.containerLeft != None and self.containerRight != None):
            self.containerLeft.configure(image=self.photo)
            self.containerLeft.image = self.photo
            self.containerRight.configure(image=self.photo)
            self.containerRight.image = self.photo
            self.histogramCanvas.get_tk_widget().grid_forget()
            subplot = 221
            self.histogramFigure = plt.Figure(figsize=(3, 3), dpi=100)
            for i, color in enumerate(['r', 'g', 'b']):
                self.histogramFigure.add_subplot(subplot).plot(cv2.calcHist(
                    [self.editedImg], [i], None, [256], [0, 256]), color=color)
                subplot = subplot + 1
            self.histogramCanvas = FigureCanvasTkAgg(
                self.histogramFigure, self.window)
            self.histogramCanvas.get_tk_widget().grid(row=1, column=0, sticky=NW)
        else:
            self.containerLeft = Label(self.window, image=self.photo)
            self.containerLeft.image = self.photo
            self.containerLeft.grid(row=0, column=0, sticky=NW)
            self.containerRight = Label(self.window, image=self.photo)
            self.containerRight.image = self.photo
            self.containerRight.grid(row=0, column=1, sticky=NW)
            self.histogramFigure = plt.figure(figsize=(3, 3), dpi=100)
            self.histogramCanvas = FigureCanvasTkAgg(
                self.histogramFigure, self.window)
            subplot = 221
            self.histogramFigure = plt.figure(figsize=(3, 3), dpi=100)
            for i, color in enumerate(['r', 'g', 'b']):
                self.histogramFigure.add_subplot(subplot).plot(cv2.calcHist(
                    [self.editedImg], [i], None, [256], [0, 256]), color=color)
                subplot = subplot + 1
            self.histogramCanvas = FigureCanvasTkAgg(
                self.histogramFigure, self.window)
            self.histogramCanvas.get_tk_widget().grid(row=1, column=0, sticky=NW)

    def setHistogram(self, img):
        self.histogramCanvas.get_tk_widget().grid_forget()
        subplot = 221
        self.histogramFigure = plt.Figure(figsize=(3, 3), dpi=100)
        for i, color in enumerate(['r', 'g', 'b']):
            self.histogramFigure.add_subplot(subplot).plot(cv2.calcHist(
                [img], [i], None, [256], [0, 256]), color=color)
            subplot = subplot + 1
        self.histogramCanvas = FigureCanvasTkAgg(
            self.histogramFigure, self.window)
        self.histogramCanvas.get_tk_widget().grid(row=1, column=0, sticky=NW)

    def addColorValue(self, theColor, addValue=10):
        self.height, self.width, no_channels = self.editedImg.shape
        lim = 255
        self.editedImg[:, :, theColor] = np.clip(
            self.editedImg[:, :, theColor]+addValue, 0, lim)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def subsColorValue(self, theColor, subsValue=10):
        self.height, self.width, no_channels = self.editedImg.shape
        lim = 0
        self.editedImg[:, :, theColor] = np.clip(
            self.editedImg[:, :, theColor]-subsValue, lim, 255)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def downSampling(self):
        origWidth = self.editedImg.shape[1]
        origHeight = self.editedImg.shape[0]
        origDimensions = (origWidth, origHeight)
        width = int(self.editedImg.shape[1]*15/100)
        height = int(self.editedImg.shape[0]*15/100)
        dimension = (width, height)
        downSampledImg = cv2.resize(
            self.editedImg, dimension, interpolation=cv2.INTER_AREA)
        self.editedImg = cv2.resize(
            downSampledImg, origDimensions, interpolation=cv2.INTER_AREA)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def quantization(self):
        height, width = self.editedImg.shape[0], self.editedImg.shape[1]
        new_img = np.zeros((height, width, 3), np.uint8)

        #  Image quantization operation , The quantification level is 2
        for i in range(height):
            for j in range(width):
                for k in range(3):  # Correspondence BGR Three channels
                    if self.editedImg[i, j][k] < 128:
                        gray = 0
                    else:
                        gray = 129
                    new_img[i, j][k] = np.uint8(gray)
        self.editedImg = new_img
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def increase_brightness(self, value=30):
        hsv = cv2.cvtColor(self.editedImg, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        self.editedImg = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def decrease_brightness(self, value=30):
        hsv = cv2.cvtColor(self.editedImg, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 0 + value
        v[v > lim] -= value
        v[v <= lim] = 0

        final_hsv = cv2.merge((h, s, v))
        self.editedImg = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def negativeImg(self):
        self.editedImg = cv2.bitwise_not(self.editedImg)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def equalizeHistogram(self):
        channels = cv2.split(self.editedImg)
        eq_channels = []
        for ch, color in zip(channels, ['R', 'G', 'B']):
            eq_channels.append(cv2.equalizeHist(ch))
        self.editedImg = cv2.merge(eq_channels)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def lowPassFilter(self):
        kernel = np.ones((5, 5), np.float32)/25
        self.editedImg = cv2.filter2D(self.editedImg, -1, kernel)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def highPassFilter(self):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        kernel = kernel/(np.sum(kernel) if np.sum(kernel) != 0 else 1)
        self.editedImg = cv2.filter2D(self.editedImg, -1, kernel)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def bandPassFilter(self):
        kernel = np.array([[-1, -1, -1], [-1, 5, -1], [-1, -1, -1]])
        kernel = kernel/(np.sum(kernel) if np.sum(kernel) != 0 else 1)
        self.editedImg = cv2.filter2D(self.editedImg, -1, kernel)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def resetEditedImg(self):
        self.editedImg = cv2.cvtColor(cv2.imread(
            self.currentFileDir), cv2.COLOR_BGR2RGB)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)

        return resized

    def displayOnRightPanel(self, image):
        self.photo = Image.fromarray(image)
        size = [450, 450]
        self.photo.thumbnail(size, Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image=self.photo)
        self.containerRight.configure(image=self.photo)
        self.containerRight.image = self.photo


App(Tk(), "Image Manipulator", "1280x720")
