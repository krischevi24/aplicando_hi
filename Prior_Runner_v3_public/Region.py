# Region.py
class Region:
    """Define una región rectangular de la imagen."""
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def clip_region(reg, img_shape):
    """Asegura que la región esté dentro de los límites de la imagen."""
    h, w = img_shape[:2]
    reg.x = max(0, min(reg.x, w - 1))
    reg.y = max(0, min(reg.y, h - 1))
    reg.w = max(1, min(reg.w, w - reg.x))
    reg.h = max(1, min(reg.h, h - reg.y))
    return reg