# QuadTree.py
from Region import Region, clip_region

class QuadTree:
    """Implementación básica de QuadTree para partición de imagen."""
    def __init__(self, x, y, w, h, image):
        self.region = Region(x, y, w, h)
        clip_region(self.region, image.shape)
        self.image = image  # grayscale
        self.children = []
        # Criterio de parada: el lado más corto es >= 100 píxeles
        if min(self.region.w, self.region.h) >= 100:
            self.divide()

    def divide(self):
        """Divide el nodo en cuatro cuadrantes."""
        rx, ry, rw, rh = self.region.x, self.region.y, self.region.w, self.region.h
        hw = rw // 2
        hh = rh // 2
        self.children = [
            QuadTree(rx, ry, hw, hh, self.image),
            QuadTree(rx + hw, ry, rw - hw, hh, self.image),
            QuadTree(rx, ry + hh, hw, rh - hh, self.image),
            QuadTree(rx + hw, ry + hh, rw - hw, rh - hh, self.image),
        ]

def get_leaves(node, leaves):
    """Recopila las regiones no divididas (hojas)."""
    if not node.children:
        leaves.append(node.region)
    else:
        for child in node.children:
            get_leaves(child, leaves)