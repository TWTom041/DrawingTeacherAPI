import threading
import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.clock import mainthread
from kivy.lang import Builder
import numpy as np
import cv2
import base64
import time

import backend

kivy.require("2.1.0")

stylized_image = -1  # numpy image
content_image = -1
style_image = -1


def crop_center(image):
    shape = image.shape
    new_shape = min(shape[0], shape[1])
    offset_y = max(shape[0] - shape[1], 0) // 2
    offset_x = max(shape[1] - shape[0], 0) // 2
    image = image[offset_y:offset_y + new_shape, offset_x:offset_x + new_shape]
    return image


def image_style_trans():
    global stylized_image, content_image, style_image
    image = backend.make_trans(
        cv2_base64(content_image),
        cv2_base64(style_image))
    image *= 255
    image = np.rint(image).astype(np.uint8).reshape(image.shape[1:])
    stylized_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def cv2_base64(image):
    base64_str = cv2.imencode(".jpg", image)[1].tobytes()
    base64_str = base64.b64encode(base64_str)
    return base64_str


class WindowManager(ScreenManager):
    pass


class CameraPage(Screen):
    def capture(self):
        global content_image
        camera = self.ids["camera"]
        camera.play = False
        texture = camera.texture

        img = np.frombuffer(texture.pixels, dtype=np.uint8).reshape((texture.size[1], texture.size[0], 4))
        content_image = crop_center(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))

        self.parent.current = "choose_style_page"
        # self.start_background_task()


class ChooseStylePage(Screen):
    def select(self, *args):
        try:
            print(args)
            self.ids["label"].text = args[1][0]
            print(self.ids["label"].text)
        except:
            print("bad")

    @mainthread
    def goto_stylized_imagepage(self):
        global stylized_image, content_image, style_image
        self.manager.current = "stylized_image_page"
        texture = Texture.create(size=(stylized_image.shape[1], stylized_image.shape[0]))
        texture.blit_buffer(cv2.flip(stylized_image, 0).tobytes(), colorfmt="bgr", bufferfmt="ubyte")
        self.manager.get_screen("stylized_image_page").ids["stylized_image"].texture = texture
        texture = Texture.create(size=(content_image.shape[1], content_image.shape[0]))
        texture.blit_buffer(cv2.flip(content_image, 0).tobytes(), colorfmt="bgr", bufferfmt="ubyte")
        self.manager.get_screen("stylized_image_page").ids["content_image"].texture = texture
        texture = Texture.create(size=(style_image.shape[1], style_image.shape[0]))
        texture.blit_buffer(cv2.flip(style_image, 0).tobytes(), colorfmt="bgr", bufferfmt="ubyte")
        self.manager.get_screen("stylized_image_page").ids["style_image"].texture = texture

    def start_background_task(self):
        global style_image
        style_image = crop_center(cv2.imread(self.ids["label"].text))
        threading.Thread(target=self.call_function).start()

    def call_function(self, *args):
        global content_image, style_image

        image_style_trans()
        self.goto_stylized_imagepage()


class LoadingPage(Screen):
    pass


class StylizedImagePage(Screen):
    pass


kv = Builder.load_file("layout.kv")


class mainApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    mainApp().run()
