import numpy as np, sys
sys.path.insert(0, '.')
from sdf import *

SKIN, EYE, MOUTH, TEETH, BONE, WOUND, SHIRT = 0, 1, 2, 3, 4, 5, 6

def rotY(p, a):
    c, s = np.cos(a), np.sin(a); q = p.copy()
    q[..., 0] = c * p[..., 0] - s * p[..., 2]; q[..., 2] = s * p[..., 0] + c * p[..., 2]
    return q
def rotZ(p, a):
    c, s = np.cos(a), np.sin(a); q = p.copy()
    q[..., 0] = c * p[..., 0] - s * p[..., 1]; q[..., 1] = s * p[..., 0] + c * p[..., 1]
    return q

def scene(p):
    p = rotZ(p, 0.12)                     # przechylona głowa
    q = mirror_x(p)
    # czaszka + szczęka + kości policzkowe
    d = ellipsoid(p, (0, 0.2, -0.05), (0.7, 0.75, 0.75))
    d = smin(d, ellipsoid(p, (0, -0.35, 0.12), (0.5, 0.45, 0.55)), 0.25)
    d = smin(d, ellipsoid(q, (0.38, -0.05, 0.45), (0.2, 0.15, 0.2)), 0.12)
    # łuk brwiowy
    d = smin(d, capsule(q, (0.08, 0.38, 0.62), (0.45, 0.42, 0.5), 0.09), 0.08)
    # zapadnięte policzki
    d = smax(d, -sphere(q, (0.5, -0.3, 0.5), 0.17), 0.1)
    # nos + nozdrza
    d = smin(d, capsule(p, (0, 0.25, 0.75), (0, -0.02, 0.86), 0.07), 0.06)
    d = smax(d, -sphere(q, (0.05, -0.04, 0.88), 0.035), 0.01)
    # uszy (lewe naderwane)
    d = smin(d, ellipsoid(p, (0.7, 0.05, -0.05), (0.1, 0.2, 0.1)), 0.05)
    d = smin(d, ellipsoid(p, (-0.7, -0.02, -0.05), (0.1, 0.12, 0.1)), 0.05)
    # szyja
    d = smin(d, capsule(p, (0, -0.6, -0.1), (0, -1.4, -0.15), 0.3), 0.15)
    # oczodoły, asymetryczne
    sockL = sphere(p, (-0.27, 0.2, 0.62), 0.16)
    sockR = ellipsoid(p, (0.27, 0.15, 0.62), (0.17, 0.12, 0.16))
    d = smax(d, -smin(sockL, sockR, 0.01), 0.04)
    eyeL = sphere(p, (-0.27, 0.19, 0.55), 0.07)
    eyeR = sphere(p, (0.29, 0.13, 0.55), 0.05)
    eyes = np.minimum(eyeL, eyeR)
    # otwarte usta
    mcut = ellipsoid(p, (0.04, -0.42, 0.62), (0.28, 0.13, 0.25))
    d = smax(d, -mcut, 0.04)
    mouth = ellipsoid(p, (0.04, -0.42, 0.45), (0.27, 0.12, 0.1))
    # zęby: górny rząd, krzywe
    teeth = 1e9 * np.ones(p.shape[:-1])
    for i, x in enumerate(np.linspace(-0.18, 0.2, 5)):
        if i == 3: continue                     # brakujący ząb
        h = 0.05 + 0.02 * ((i * 7) % 3)
        teeth = np.minimum(teeth, box(p, (x, -0.33 - h / 2, 0.6), (0.03, h / 2, 0.03), 0.01))
    for x in (-0.1, 0.05, 0.16):
        teeth = np.minimum(teeth, box(p, (x, -0.51, 0.58), (0.03, 0.03, 0.03), 0.01))
    # odsłonięta czaszka (płat na górze z lewej)
    bone_reg = sphere(p, (-0.35, 0.65, 0.35), 0.32)
    skin_d = smax(d, -bone_reg, 0.02)
    bone = smax(d - 0.0, bone_reg, 0.02) + 0.03   # kość lekko wgłębiona
    # rana na policzku
    wound_reg = capsule(p, (0.3, -0.05, 0.6), (0.48, -0.25, 0.48), 0.05)
    # koszula
    shirt = smin(ellipsoid(p, (0, -1.35, -0.15), (0.75, 0.25, 0.45)),
                 capsule(p, (0, -1.25, -0.15), (0, -1.6, -0.15), 0.38), 0.1)
    shirt = smax(shirt, -capsule(p, (0, -1.05, 0.05), (0, -1.4, 0.15), 0.18), 0.05)

    parts = [skin_d, eyes, mouth, teeth, bone, shirt]
    ids = [SKIN, EYE, MOUTH, TEETH, BONE, SHIRT]
    dist = parts[0]; mat = np.full(dist.shape, SKIN)
    for dd, mid in zip(parts[1:], ids[1:]):
        mat = np.where(dd < dist, mid, mat); dist = np.minimum(dist, dd)
    mat = np.where((mat == SKIN) & (wound_reg < 0.02), WOUND, mat)
    return dist, mat

if __name__ == '__main__':
    g = render(scene, cam=(1.1, 0.2, 3.0), target=(0, -0.15, 0), fov=0.95)
    # hue-shift: cienie w fiolet, światła w żółć
    pal = {SKIN:  [(38, 34, 56), (64, 82, 70), (104, 128, 86), (156, 172, 112)],
           EYE:   [(250, 238, 150)],
           MOUTH: [(30, 10, 22), (52, 16, 30)],
           TEETH: [(120, 104, 80), (170, 155, 110), (214, 200, 150), (230, 220, 175)],
           BONE:  [(90, 80, 84), (150, 140, 130), (200, 190, 170), (232, 226, 205)],
           WOUND: [(50, 12, 24), (96, 24, 36), (140, 40, 44), (170, 60, 56)],
           SHIRT: [(30, 26, 44), (56, 48, 70), (86, 74, 96), (110, 98, 118)]}
    big, small = pixelize(g, pal, bg=(22, 20, 34), outline=(14, 10, 20), emissive=(EYE,))
    big.save('./v2.png')
