from collections import deque
import numpy as np
import pandas as pd
from Paint import PaintColor, LightColor, Paint
from copy import deepcopy
import cv2
from light_scanning import analyze


class Tile:
    def __init__(self, top_paint=None, right_paint=None, bottom_paint=None, left_paint=None, like=None, light_color=None):
        if like is not None:
            assert isinstance(like, Tile)
            self.sides = deepcopy(like.sides)
        else:
            assert top_paint is not None
            assert right_paint is not None
            assert bottom_paint is not None
            assert left_paint is not None
            self.sides = deque([top_paint, right_paint, bottom_paint, left_paint])
        if light_color is not None:
            self.update_light(light_color)

    def rotate(self, amount_clockwise=1):
        self.sides.rotate(amount_clockwise)

    def update_light(self, light_color):
        assert isinstance(light_color, LightColor)
        for side in self.sides:
            assert isinstance(side, Paint)
            side.light_color = light_color

    def to_pixels(self, side_length=301):
        # stolen from here
        # https://stackoverflow.com/questions/51875114/triangle-filling-in-opencv
        image = np.zeros((side_length, side_length, 3), np.uint8)
        pt1 = (0, 0)
        pt2 = (side_length, 0)
        pt3 = (side_length // 2, side_length // 2)
        triangle_cnt = np.array([pt1, pt2, pt3])

        for _ in range(4):
            color = self.sides[0].RGB.tolist()[::-1]
            cv2.drawContours(image, [triangle_cnt], 0, color, -1)
            self.rotate()
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image


def test_tiles():
    paired_light_data = analyze()
    # Let's drop some rows to clean up the data
    for key in paired_light_data.keys():
        assert isinstance(paired_light_data[key], pd.DataFrame)
        paired_light_data[key] = paired_light_data[key].loc[
            # (paired_light_data[key]['min difference'] < 20) &
            (paired_light_data[key]['color difference'] > 50)]
        l_color_data = paired_light_data[key].loc[
            (paired_light_data[key]['light where same'] == key[0])]
        r_color_data = paired_light_data[key].loc[
            (paired_light_data[key]['light where same'] == key[1])]
        print()

    m1 = Tile(
        top_paint=Paint(
            paint_color=PaintColor.Magenta),
        right_paint=Paint(
            paint_color=PaintColor.Taupe),
        bottom_paint=Paint(
            paint_color=PaintColor.Raspberry),
        left_paint=Paint(
            paint_color=PaintColor.Lime_Green))
    m2 = Tile(
        top_paint=Paint(
            paint_color=PaintColor.Red_Orange),
        right_paint=Paint(
            paint_color=PaintColor.Golden_Yellow),
        bottom_paint=Paint(
            paint_color=PaintColor.Jade_Green),
        left_paint=Paint(
            paint_color=PaintColor.Maroon))
    for i, tile in enumerate([m1, m2]):
        for light in [LightColor.M, LightColor.C, LightColor.W]:
            tile.update_light(light)
            pixels = tile.to_pixels()
            cv2.imshow(f'tile:{i}, light_color: {light.name}', pixels)


if __name__ == '__main__':
    test_tiles()
    cv2.waitKey(0)