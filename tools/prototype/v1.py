import numpy as np, sys
sys.path.insert(0, '.')
from sdf import *

SKIN, EYE, MOUTH, TEETH = 0, 1, 2, 3

def scene(p):
    q = mirror_x(p)
    head = ellipsoid(p, (0, 0.1, 0), (0.75, 0.9, 0.8))
    jaw = ellipsoid(p, (0, -0.45, 0.15), (0.55, 0.4, 0.55))
    d = smin(head, jaw, 0.2)
    ear = ellipsoid(q, (0.75, 0.05, 0), (0.12, 0.22, 0.1))
    d = smin(d, ear, 0.05)
    sock = sphere(q, (0.3, 0.2, 0.72), 0.18)
    d = smax(d, -sock, 0.05)
    eye = sphere(q, (0.3, 0.2, 0.62), 0.12)
    mouth_cut = box(p, (0, -0.5, 0.65), (0.3, 0.06, 0.2), 0.03)
    d = smax(d, -mouth_cut, 0.03)
    mouth_in = box(p, (0, -0.5, 0.55), (0.28, 0.05, 0.05))
    dist = np.minimum(np.minimum(d, eye), mouth_in)
    mat = np.where(dist == eye, EYE, np.where(dist == mouth_in, MOUTH, SKIN))
    return dist, mat

g = render(scene)
pal = {SKIN: [(40, 60, 40), (70, 105, 60), (110, 150, 85), (160, 195, 120)],
       EYE: [(255, 230, 80)], MOUTH: [(60, 10, 20), (90, 20, 30)],
       TEETH: [(200, 190, 150)]}
big, small = pixelize(g, pal, bg=(28, 24, 40), emissive=(EYE,))
big.save('./v1.png')
