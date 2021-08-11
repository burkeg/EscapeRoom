from enum import Enum
from light_scanning import find_pigment_colors, paint_color_names, light_color_names
from itertools import chain
from CMYK import Converter

PaintColor = Enum('PaintColor', dict([(letter, i + 1) for i, letter in enumerate(chain.from_iterable(paint_color_names))]))
LightColor = Enum('LightColor', dict([(letter, i + 1) for i, letter in enumerate(light_color_names)]))

class Paint:
    _color_mapping = find_pigment_colors()
    def __init__(self, paint_color=None, light_color=None):
        self._paint_color = PaintColor.Magenta if paint_color is None else paint_color
        self._light_color = LightColor.W if light_color is None else light_color
        self._rgb_filter = [255, 255, 255]
        self._cmyk_filter = [1, 1, 1, 1]
        assert isinstance(self.paint_color, PaintColor)
        assert isinstance(self.light_color, LightColor)

    @property
    def RGB(self):
        rgb = self._color_mapping.loc[
                  (self._color_mapping['Light Color'] == self.light_color.name) &
                  (self._color_mapping['Paint Color'] == self.paint_color.name)]['Average'].values[0]
        return self._apply_filters(*rgb)

    @property
    def CMYK(self):
        return Converter.rgb_to_cmyk(*self.RGB)

    @property
    def paint_color(self):
        return self._paint_color

    @property
    def light_color(self):
        return self._light_color

    @light_color.setter
    def light_color(self, new_val):
        assert isinstance(new_val, LightColor)
        self._light_color = new_val

    @property
    def rgb_filter(self):
        return self._rgb_filter

    @rgb_filter.setter
    def rgb_filter(self, *args):
        assert len(args) == 3
        self._rgb_filter = [*args]

    @property
    def cmyk_filter(self):
        return self._cmyk_filter

    @cmyk_filter.setter
    def cmyk_filter(self, *args):
        assert len(args) == 4
        self._cmyk_filter = [*args]

    def _apply_filters(self, *rgb):
        assert len(rgb) == 3
        rgb_inter = list(map(lambda pair: int((pair[0] / 255) * pair[1]), zip(self._rgb_filter, rgb)))
        cmyk = Converter.rgb_to_cmyk(*rgb_inter)
        cmyk_inter = list(map(lambda pair: pair[0] * pair[1], zip(self._cmyk_filter, cmyk)))
        return Converter.cmyk_to_rgb(*cmyk_inter)


if __name__ == '__main__':
    pnt = Paint(paint_color=PaintColor.Blue, light_color=LightColor.W)
    print('original rgb', pnt.RGB)
    pnt.cmyk_filter[1] = 0
    print('rgb after removing all magenta', pnt.RGB)
    # print('rgb after removing all magenta', Converter.cmyk_to_rgb(*cmyk))
