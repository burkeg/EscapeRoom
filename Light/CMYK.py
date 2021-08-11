class Converter:
    # https://www.rapidtables.com/convert/color/rgb-to-cmyk.html
    @staticmethod
    def rgb_to_cmyk(r, g, b):
        rp = r / 255
        gp = g / 255
        bp = b / 255
        k = 1 - max(rp, gp, bp)
        c = (1 - rp - k) / (1 - k)
        m = (1 - gp - k) / (1 - k)
        y = (1 - bp - k) / (1 - k)
        return [c, m, y, k]


    # https://www.rapidtables.com/convert/color/cmyk-to-rgb.html
    @staticmethod
    def cmyk_to_rgb(c, m, y, k):
        r = 255 * (1 - c) * (1 - k)
        g = 255 * (1 - m) * (1 - k)
        b = 255 * (1 - y) * (1 - k)
        return list(map(lambda x: int(x), [r, g, b]))


    @staticmethod
    def bgr_to_rgb(b, g, r):
        return [r, g, b]


    @staticmethod
    def rgb_to_bgr(r, g, b):
        return [b, g, r]
