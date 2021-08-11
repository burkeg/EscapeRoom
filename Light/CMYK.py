class Converter:
    # https://www.rapidtables.com/convert/color/rgb-to-cmyk.html
    @staticmethod
    def rgb_to_cmyk(r, g, b):
        rp = r / 255
        gp = g / 255
        bp = b / 255
        k = 1 - max(rp, gp, bp)
        c = (1 - rp - k) / (1 - k) if k != 1 else 0
        m = (1 - gp - k) / (1 - k) if k != 1 else 0
        y = (1 - bp - k) / (1 - k) if k != 1 else 0
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


    # Yes I know they're the same method but it's probably faster to rewrite it than
    # call the other one
    @staticmethod
    def rgb_to_bgr(r, g, b):
        return [b, g, r]

    @staticmethod
    def filter_rgb_by_rgb(r, g, b, r_filt, g_filt, b_filt):
        return list(map(lambda pair: int((pair[0] / 255) * pair[1]),
                          zip([r, g, b], [r_filt, g_filt, b_filt])))

    @staticmethod
    def filter_cmyk_by_cmyk(c, m, y, k, c_filt, m_filt, y_filt, k_filt):
        return list(map(lambda pair: pair[0] * pair[1],
                        zip([c, m, y, k], [c_filt, m_filt, y_filt, k_filt])))

    @staticmethod
    def filter_rgb_by_cmyk(r, g, b, c_filt, m_filt, y_filt, k_filt):
        return Converter.cmyk_to_rgb(
            *Converter.filter_cmyk_by_cmyk(
                *Converter.rgb_to_cmyk(r, g, b),
                c_filt, m_filt, y_filt, k_filt))

    @staticmethod
    def filter_cmyk_by_rgb(c, m, y, k, r_filt, g_filt, b_filt):
        return Converter.rgb_to_cmyk(
            *Converter.filter_rgb_by_rgb(
                *Converter.cmyk_to_rgb(c, m, y, k),
                r_filt, g_filt, b_filt))

