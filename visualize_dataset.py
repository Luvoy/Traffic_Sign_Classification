import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
from torchvision.transforms import transforms

import data_set_loader


class DataVisualizer(object):
    def __init__(self, data_set, switch_method="RANDOM"):

        self.data_set = data_set
        self.size = len(data_set)

        self.switch_method = switch_method
        if switch_method == "NATURAL":
            self.index = 0
        elif switch_method == "RANDOM":
            self.index = random.randint(0, self.size - 1)
            self.history_index_stack = []

        plt.subplots_adjust(bottom=0.2)

        self.ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])

        self.button_next = Button(self.ax_next, 'Next')
        self.button_next.on_clicked(self.next_img)

        self.button_prev = Button(self.ax_prev, 'Previous')
        self.button_prev.on_clicked(self.prev_img)

        self.ax_img = plt.axes()

    def show_img(self):
        plt.cla()

        # override here when you use it in another project
        self.ax_img.set_title(
            f"index: {self.index} shape: {self.data_set[self.index]['image'].shape} class_id: {self.data_set[self.index]['class_id']}"
        )
        # self.ax_img.imshow(self.data_set[self.index]['image'][0], cmap='gray') # cv2 style

        self.ax_img.imshow(transforms.ToPILImage()(
            self.data_set[self.index]['image']))

    def next_img(self, fuck_bug):
        self.history_index_stack.append(self.index)
        if self.switch_method == "NATURAL":
            if self.index < self.size - 1:
                self.index += 1
                self.show_img()
        elif self.switch_method == "RANDOM":
            self.index = random.randint(0, self.size - 1)
            self.show_img()
        else:
            pass

    def prev_img(self, fuck_bug):
        if self.switch_method == "NATURAL":
            if self.index > 0:
                self.index -= 1
                self.show_img()
        elif self.switch_method == "RANDOM":
            if self.history_index_stack:
                self.index = self.history_index_stack.pop()
                self.show_img()
        else:
            pass

    def run(self):
        self.show_img()
        plt.show()


if __name__ == "__main__":
    vr = DataVisualizer(data_set_loader.train_dataset, switch_method="RANDOM")
    vr.run()
