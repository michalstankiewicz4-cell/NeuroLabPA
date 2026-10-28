import numpy as np
from PIL import Image

# ---------- SDF prymitywy ----------
def length(v): return np.sqrt((v**2).sum(-1))
def sphere(p, c, r): return length(p - np.array(c)) - r
def ellipsoid(p, c, r):
    q = (p - np.array(c)) / np.array(r)
    k0 = length(q); k1 = length(q / np.array(r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)
def box(p, c, b, r=0.0):
    q = np.abs(p - np.array(c)) - np.array(b) + r
    return length(np.maximum(q, 0)) + np.minimum(q.max(-1), 0) - r
def capsule(p, a, b, r):
    a = np.array(a); b = np.array(b)
    pa = p - a; ba = b - a
    h = np.clip((pa * ba).sum(-1) / (ba * ba).sum(), 0, 1)
    return length(pa - ba * h[..., None]) - r
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)
def smax(a, b, k): return -smin(-a, -b, k)

def mirror_x(p):
    q = p.copy(); q[..., 0] = np.abs(q[..., 0]); return q

# ---------- raymarcher ----------
def render(scene, W=64, H=64, cam=(0, 0.1, 3.2), target=(0, 0, 0), fov=0.9,
           light=(-0.6, 0.7, 0.8), steps=96):
    cam = np.array(cam, float); target = np.array(target, float)
    fw = target - cam; fw /= np.linalg.norm(fw)
    rt = np.cross(fw, [0, 1, 0]); rt /= np.linalg.norm(rt)
    up = np.cross(rt, fw)
    ys, xs = np.mgrid[0:H, 0:W]
    u = (xs + 0.5) / W * 2 - 1; v = -((ys + 0.5) / H * 2 - 1)
    d = fw + (u[..., None] * rt + v[..., None] * up) * np.tan(fov / 2)
    d /= length(d)[..., None]
    t = np.zeros((H, W)); hit = np.zeros((H, W), bool)
    for _ in range(steps):
        p = cam + d * t[..., None]
        dist, _m = scene(p)
        act = ~hit & (t < 8)
        hit |= act & (dist < 1e-3)
        t = np.where(act & ~hit, t + dist * 0.9, t)
    p = cam + d * t[..., None]
    _, mat = scene(p)
    e = 1e-3
    n = np.stack([scene(p + np.array(o))[0] - scene(p - np.array(o))[0]
                  for o in ([e, 0, 0], [0, e, 0], [0, 0, e])], -1)
    n /= np.maximum(length(n), 1e-9)[..., None]
    L = np.array(light, float); L /= np.linalg.norm(L)
    diff = np.clip((n * L).sum(-1), 0, 1)
    # miękki cień
    sh = np.ones((H, W)); ts = np.full((H, W), 0.02)
    for _ in range(32):
        dd, _m = scene(p + n * 2e-3 + L * ts[..., None])
        sh = np.minimum(sh, np.clip(8 * dd / ts, 0, 1))
        ts += np.clip(dd, 0.01, 0.2)
    # ambient occlusion
    ao = np.ones((H, W))
    for i in range(1, 6):
        h = 0.04 * i
        dd, _m = scene(p + n * h)
        ao -= (h - dd) * (0.5 ** i) * 4
    ao = np.clip(ao, 0, 1)
    rim = np.clip(1 + (n * d).sum(-1), 0, 1) ** 3
    return dict(hit=hit, mat=np.where(hit, mat, -1), depth=t, n=n,
                diff=diff * sh, ao=ao, rim=rim)

# ---------- pixel-art pass ----------
def pixelize(g, palettes, bg, outline=(20, 12, 24), bands=(0.18, 0.45, 0.75),
             scale=8, emissive=()):
    H, W = g['hit'].shape
    light = 0.15 + 0.85 * g['diff'] * (0.5 + 0.5 * g['ao'])
    light = light + 0.25 * g['rim']
    idx = np.digitize(light, bands)  # 0..3
    img = np.zeros((H, W, 3), np.uint8); img[:] = bg
    for m, pal in palettes.items():
        sel = g['mat'] == m
        pal = np.array(pal, np.uint8)
        ii = np.full_like(idx, len(pal) - 1) if m in emissive else np.clip(idx, 0, len(pal) - 1)
        img[sel] = pal[ii[sel]]
    # kontur: krawędź sylwetki + granice materiałów + skoki głębi
    mat = g['mat']; dep = np.where(g['hit'], g['depth'], 99)
    edge = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        m2 = np.roll(np.roll(mat, dy, 0), dx, 1)
        d2 = np.roll(np.roll(dep, dy, 0), dx, 1)
        edge |= g['hit'] & ((m2 == -1) | (d2 - dep > 0.25))
    for dy, dx in ((0, 1), (1, 0)):
        m2 = np.roll(np.roll(mat, dy, 0), dx, 1)
        b = g['hit'] & (m2 != mat) & (m2 != -1)
        # rysuj tylko po stronie jaśniejszego materiału = ciemniejszy z dwóch
        edge |= b & np.isin(mat, [k for k in palettes if k not in emissive])
    img[edge] = outline
    big = Image.fromarray(img).resize((W * scale, H * scale), Image.NEAREST)
    return big, Image.fromarray(img)
