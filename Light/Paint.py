from enum import Enum
from light_scanning import find_pigment_colors, paint_color_names, light_color_names
from itertools import chain

PaintColor = Enum('PaintColor', dict([(letter, i + 1) for i, letter in enumerate(chain.from_iterable(paint_color_names))]))
LightColor = Enum('LightColor', dict([(letter, i + 1) for i, letter in enumerate(light_color_names)]))

class Paint:
    _color_mapping = find_pigment_colors()
    def __init__(self, paint_color=None, light_color=None):
        self._paint_color = PaintColor.Magenta if paint_color is None else paint_color
        self._light_color = LightColor.W if light_color is None else light_color
        assert isinstance(self.paint_color, PaintColor)
        assert isinstance(self.light_color, LightColor)

    @property
    def RGB(self):
        return self._color_mapping.loc[
                  (self._color_mapping['Light Color'] == self.light_color.name) &
                  (self._color_mapping['Paint Color'] == self.paint_color.name)]['Average'].values[0]

    @property
    def light_color(self):
        return self._light_color

    @property
    def paint_color(self):
        return self._paint_color

    @light_color.setter
    def light_color(self, new_val):
        assert isinstance(new_val, LightColor)
        self._light_color = new_val




if __name__ == '__main__':
    pnt = Paint(paint_color=PaintColor.Blue, light_color=LightColor.W)
    print(pnt.RGB)