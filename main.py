from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter.ttk import Progressbar, Label
from PIL import ImageGrab, Image, ImageOps
import numpy as np
import torch
from torch import nn

class Paint(object):

    DEFAULT_PEN_SIZE = 50.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='Карандаш', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.color_button = Button(self.root, text='Цвет', command=self.choose_color)
        self.color_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='Ластик', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.brush_button = Button(self.root, text='Очистить', command=self.clear)
        self.brush_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=40, to=60, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(rowspan=16, columnspan=5)

        self.progress = []
        for i in range(10):
            label = Label(text=f"{i}")
            label.grid(row=4+i, column=7, padx = 10)

            a = Progressbar(self.root, orient=HORIZONTAL, length=100, mode='determinate', maximum=100, value=0)
            a.grid(row=4+i, column=6, padx = 10)

            self.progress.append(a)

        self.setup()
        self.model = self.load_model()  # Загрузка модели
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
        self.predict_digit()

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear(self):
        self.c.delete("all")
        for bar in self.progress:
            bar['value'] = 0

    def save_image(self):
        x = self.root.winfo_rootx() + self.c.winfo_x()
        y = self.root.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save("drawing.png")

    def predict_digit(self):
        x = self.root.winfo_rootx() + self.c.winfo_x()
        y = self.root.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1))

        # Preprocess the image for the model
        image = image.convert('L')  # Convert to grayscale
        image = ImageOps.invert(image)  # Invert the image
        image = image.resize((28, 28))  # Resize to 28x28 pixels

        # Ensure image has 1 channel
        if image.mode != 'L':
            image = image.convert('L')

        image = np.array(image)  # Convert PIL image to numpy array
        image = image / 255.0  # Normalize the image
        image = image.reshape(1, 28, 28, 1)  # Batch size 1, 1 channel, 28x28 size

        # Convert numpy array to torch tensor
        image_tensor = torch.from_numpy(image).float().permute(0, 3, 1, 2)

        # Perform prediction
        with torch.no_grad():
            self.model.eval()  # Set model to evaluation mode
            outputs = self.model(image_tensor)  # Forward pass
            probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities

        probabilities = probabilities.numpy()  # Convert probabilities to numpy array
        self.update_progress(probabilities)

    def update_progress(self, probabilities):
        for i, bar in enumerate(self.progress):
            bar['value'] = probabilities[0][i] * 100

    def load_model(self):
        device = torch.device("cpu")
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(4 * 4 * 64, 520),

            nn.ReLU(),
            nn.Linear(520, 256),

            nn.ReLU(),
            nn.Linear(256, 10)
        )  # Assuming Model is your custom model inheriting Sequential
        model.load_state_dict(torch.load('model.pt', map_location=device).state_dict())
        model.to(device)
        model.eval()
        return model


if __name__ == '__main__':
    Paint()
