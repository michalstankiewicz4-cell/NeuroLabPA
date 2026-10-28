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
    d = smin(d, capsule(p, (0, -0.6, -0.1), (0, -1.3, -0.15), 0.25), 0.15)
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
        teeth = np.minimum(teeth, box(p, (x, -0.33 - h / 2, 0.7), (0.035, h / 2, 0.04), 0.01))
    for x in (-0.1, 0.05, 0.16):
        teeth = np.minimum(teeth, box(p, (x, -0.52, 0.68), (0.035, 0.035, 0.04), 0.01))
    # odsłonięta czaszka (płat na górze z lewej)
    bone_reg = sphere(p, (0.25, 0.62, 0.4), 0.3) + 0.06*np.sin(14*p[...,0])*np.sin(11*p[...,2]+1.3)
    skin_d = smax(d, -bone_reg, 0.02)
    bone = smax(d + 0.04, bone_reg, 0.02) + 0.012*np.sin(40*p[...,0]+30*p[...,1])
    crack = capsule(p, (0.15, 0.85, 0.45), (0.35, 0.55, 0.62), 0.025)   # kość lekko wgłębiona
    # rana na policzku
    wound_reg = capsule(p, (0.62, 0.2, 0.3), (0.45, -0.3, 0.55), 0.07)
    # koszula
    shirt = smin(capsule(p, (-0.75, -1.35, -0.2), (0.75, -1.35, -0.2), 0.28),
                 ellipsoid(p, (0, -1.8, -0.15), (0.9, 0.6, 0.45)), 0.2)
    shirt = smax(shirt, -capsule(p, (0, -1.05, 0.05), (0, -1.4, 0.15), 0.18), 0.05)

    parts = [skin_d, eyes, mouth, teeth, bone, shirt]
    ids = [SKIN, EYE, MOUTH, TEETH, BONE, SHIRT]
    dist = parts[0]; mat = np.full(dist.shape, SKIN)
    for dd, mid in zip(parts[1:], ids[1:]):
        mat = np.where(dd < dist, mid, mat); dist = np.minimum(dist, dd)
    mat = np.where((mat == SKIN) & (wound_reg < 0.02), WOUND, mat)
    mat = np.where((mat == BONE) & (crack < 0), MOUTH, mat)
    return dist, mat

if __name__ == '__main__':
    g = render(scene, light=(0.7, 0.6, 0.6), cam=(0.75, 0.25, 3.1), target=(0, -0.15, 0), fov=0.95)
    # hue-shift: cienie w fiolet, światła w żółć
    pal = {SKIN:  [(38, 34, 56), (64, 82, 70), (104, 128, 86), (156, 172, 112)],
           EYE:   [(250, 238, 150)],
           MOUTH: [(30, 10, 22), (52, 16, 30)],
           TEETH: [(120, 104, 80), (170, 155, 110), (214, 200, 150), (230, 220, 175)],
           BONE:  [(80, 70, 78), (130, 120, 112), (176, 164, 140), (214, 204, 176)],
           WOUND: [(50, 12, 24), (96, 24, 36), (140, 40, 44), (170, 60, 56)],
           SHIRT: [(30, 26, 44), (56, 48, 70), (86, 74, 96), (110, 98, 118)]}
    big, small = pixelize(g, pal, bg=(22, 20, 34), outline=(14, 10, 20), emissive=(EYE,))
    big.save('./v5.png')
