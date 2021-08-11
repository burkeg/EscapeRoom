from collections import deque
import numpy as np
import pandas as pd
from Paint import PaintColor, LightColor, Paint
from copy import deepcopy
import cv2
from light_scanning import analyze, get_all_pareto_fronts
from CMYK import Converter


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

    @property
    def top(self):
        return self.sides[0]

    @top.setter
    def top(self, new_val):
        assert isinstance(new_val, Paint)
        self.sides[0] = new_val

    @property
    def right(self):
        return self.sides[1]

    @right.setter
    def right(self, new_val):
        assert isinstance(new_val, Paint)
        self.sides[1] = new_val

    @property
    def bottom(self):
        return self.sides[2]

    @bottom.setter
    def bottom(self, new_val):
        assert isinstance(new_val, Paint)
        self.sides[2] = new_val

    @property
    def left(self):
        return self.sides[3]

    @left.setter
    def left(self, new_val):
        assert isinstance(new_val, Paint)
        self.sides[3] = new_val

    def rotate(self, amount_clockwise=1):
        self.sides.rotate(amount_clockwise)

    def update_light(self, light_color):
        assert isinstance(light_color, LightColor)
        for side in self.sides:
            assert isinstance(side, Paint)
            side.light_color = light_color

    def to_pixels(self, side_length=31, with_text=False):
        # stolen from here
        # https://stackoverflow.com/questions/51875114/triangle-filling-in-opencv
        image = np.zeros((side_length, side_length, 3), np.uint8)
        pt1 = (0, 0)
        pt2 = (side_length, 0)
        pt3 = (side_length // 2, side_length // 2)
        triangle_cnt = np.array([pt1, pt2, pt3])

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (side_length // 4, side_length // 8)
        fontScale = 0.4 * side_length / 301
        fontColor = (255, 255, 255)
        lineType = 0

        for _ in range(4):
            paint = self.sides[0]
            assert isinstance(paint, Paint)
            color = Converter.rgb_to_bgr(*paint.RGB)
            cv2.drawContours(image, [triangle_cnt], 0, color, -1)
            self.rotate()
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        if with_text:
            for _ in range(4):
                paint = self.sides[0]
                assert isinstance(paint, Paint)
                color = Converter.rgb_to_bgr(*paint.RGB)
                cmyk_str = '[' + ','.join([f'{a:.2f}' for a in paint.CMYK]) + ']'
                cmyk_filter_str = '[' + ','.join([f'{a:.2f}' for a in paint.cmyk_filter]) + ']'
                message = f'Light:{paint.light_color.name}\n' \
                          f'Paint:{paint.paint_color.name}\n' \
                          f'rgb:{paint.RGB}\n' \
                          f'cmyk:{cmyk_str}\n' \
                          f'rgb_filter:{paint.rgb_filter}\n' \
                          f'cmyk_filter:{cmyk_filter_str}'
                for i, line in enumerate(message.split('\n')):
                    text_size = cv2.getTextSize(message, font, fontScale, 1)
                    bottomLeftCornerOfText = (side_length // 4,
                                              int(text_size[0][1] * (1.3 * (i + 1))))
                    cv2.putText(image, line,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)
                self.rotate()
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image


def test_tiles():
    paired_light_data = analyze()
    pareto_fronts = get_all_pareto_fronts(paired_light_data)

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

def demo_pareto_fronts():
    paired_light_data = analyze()
    pareto_fronts = get_all_pareto_fronts(paired_light_data)
    for light_color_pair in pareto_fronts.keys():
        for i, main_light_color in enumerate(light_color_pair):
            other_light_color = light_color_pair[1-i]
            # if main_light_color != 'C' or i != 0:
            #     continue
            pareto_front = pareto_fronts[light_color_pair][i]
            tile_dict = dict()
            for row_index, row in pareto_front.iterrows():
                paint_1, paint_2, _, diff_in_other_light, _, diff_in_curr_light = row.values
                tile = Tile(
                    top_paint=Paint(
                        paint_color=PaintColor[paint_1],
                        light_color=LightColor[main_light_color]),
                    right_paint=Paint(
                        paint_color=PaintColor[paint_2],
                        light_color=LightColor[main_light_color]),
                    bottom_paint=Paint(
                        paint_color=PaintColor[paint_2],
                        light_color=LightColor[other_light_color]),
                    left_paint=Paint(
                        paint_color=PaintColor[paint_1],
                        light_color=LightColor[other_light_color]))
                tile_dict[(diff_in_other_light, diff_in_curr_light)] = tile
            stats = np.array([[a, b] for a, b in tile_dict.keys()])
            max_diff_in_other_light, max_diff_in_curr_light = np.max(stats, 0)
            min_diff_in_other_light, min_diff_in_curr_light = np.min(stats, 0)
            scale_factor_in_other_light = max_diff_in_other_light - min_diff_in_other_light
            scale_factor_in_curr_light = max_diff_in_curr_light - min_diff_in_curr_light
            normalized_tile_dict = dict()
            for (orig_other_diff, orig_curr_diff), tile in tile_dict.items():
                normalized_tile_dict[(
                    (orig_other_diff - min_diff_in_other_light) / scale_factor_in_other_light,
                    (orig_curr_diff - min_diff_in_curr_light) / scale_factor_in_curr_light,
                )] = tile

            # Now the tiles are colored correctly and have their coordinates normalized.

            # Stolen from here:
            # https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
            height, width = 3000, 3000
            tile_size = 301
            image = np.zeros((height, width, 3), np.uint8)
            l_img = image
            for (x_pos, y_pos), tile in normalized_tile_dict.items():
                s_img = tile.to_pixels(tile_size, with_text=True)
                x_offset = int(tile_size + (width - 2*tile_size)*x_pos)
                y_offset = int(tile_size + (height - 2*tile_size)*y_pos)
                l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img
            cv2.imwrite(f'./Pareto Fronts/main light {main_light_color}_____secondary light {other_light_color}.jpg', l_img)
            cv2.waitKey(0)
            print()
    print()

def demo_cmyk_tiles():
    top_paint = Paint(
        paint_color=PaintColor['White'],
        light_color=LightColor['W'])
    top_paint.rgb_filter = [0, 255, 255]
    right_paint = Paint(
        paint_color=PaintColor['White'],
        light_color=LightColor['W'])
    right_paint.rgb_filter = [255, 0, 255]
    bottom_paint = Paint(
        paint_color=PaintColor['White'],
        light_color=LightColor['W'])
    bottom_paint.rgb_filter = [255, 255, 0]
    left_paint = Paint(
        paint_color=PaintColor['White'],
        light_color=LightColor['W'])
    left_paint.rgb_filter = [255, 255, 255]
    tile1 = Tile(top_paint, right_paint, bottom_paint, left_paint)
    cv2.imshow('demo cmyk', tile1.to_pixels(1001, with_text=True))

    top_paint = Paint(
        paint_color=PaintColor['White'],
        light_color=LightColor['W'])
    top_paint.cmyk_filter = [0, 1, 1, 1]
    right_paint = Paint(
        paint_color=PaintColor['White'],
        light_color=LightColor['W'])
    right_paint.cmyk_filter = [1, 0, 1, 1]
    bottom_paint = Paint(
        paint_color=PaintColor['White'],
        light_color=LightColor['C'])
    bottom_paint.cmyk_filter = [1, 1, 1, 1]
    left_paint = Paint(
        paint_color=PaintColor['White'],
        light_color=LightColor['W'])
    left_paint.cmyk_filter = [1, 1, 1, 1]
    tile2 = Tile(top_paint, right_paint, bottom_paint, left_paint)
    cv2.imshow('demo cmyk 2', tile2.to_pixels(1001, with_text=True))

    top_paint = Paint(
        paint_color=PaintColor['Perfect_Cyan'],
        light_color=LightColor['R'])
    top_paint.rgb_filter = [255, 255, 255]
    top_paint.cmyk_filter = [1, 1, 1, 1]
    right_paint = Paint(
        paint_color=PaintColor['Perfect_Cyan'],
        light_color=LightColor['G'])
    right_paint.rgb_filter = [255, 255, 255]
    right_paint.cmyk_filter = [1, 1, 1, 1]
    bottom_paint = Paint(
        paint_color=PaintColor['Perfect_Cyan'],
        light_color=LightColor['B'])
    bottom_paint.rgb_filter = [255, 255, 255]
    bottom_paint.cmyk_filter = [1, 1, 1, 1]
    left_paint = Paint(
        paint_color=PaintColor['Perfect_Blue'],
        light_color=LightColor['W'])
    left_paint.rgb_filter = [255, 255, 255]
    left_paint.cmyk_filter = [1, 1, 1, 1]
    tile3 = Tile(top_paint, right_paint, bottom_paint, left_paint)
    cv2.imshow('demo cmyk 2', tile3.to_pixels(301, with_text=True))




if __name__ == '__main__':
    demo_pareto_fronts()
    cv2.waitKey(0)