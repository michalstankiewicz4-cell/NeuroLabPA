#!/usr/bin/env python3
"""Laboratorium obrazów · dane z Quick, Draw! (Google, CC BY 4.0) → img/data/.

Dla każdej kategorii pobiera tylko początek pliku numpy_bitmap (rysunki 28×28, 1 bajt na piksel):
nagłówek .npy i pierwsze N rysunków, przez zapytanie o fragment pliku (HTTP Range). Rysunki pomniejsza
do 16×16 (średnia z obszaru), przypisuje do 16 odcieni (0 = tło, 15 = kreska) i zapisuje jako jeden
obrazek PNG na kategorię: siatka 50 kolumn, rysunek obok rysunku. Te pliki widać w folderze i w nich
dokładnie to, z czego uczy się sieć.

Uruchomienie (z katalogu repo):  python3 img/tools/build_quickdraw.py
Wymaga: numpy, pillow.
"""
import json, os, sys, urllib.request
import numpy as np
from PIL import Image

N = 2000            # rysunków na kategorię
SIZE = 16           # bok obrazka po pomniejszeniu
LEVELS = 16         # liczba odcieni
COLS = 50           # rysunków w wierszu pliku PNG
URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy'
# (nazwa w Quick, Draw!, nazwa polska)
CATS = [('cat', 'kot'), ('house', 'dom'), ('tree', 'drzewo'), ('fish', 'ryba'), ('car', 'samochód'),
        ('sun', 'słońce'), ('flower', 'kwiat'), ('bicycle', 'rower'), ('apple', 'jabłko'),
        ('star', 'gwiazda'), ('skull', 'czaszka'), ('moon', 'księżyc')]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
UA = {'User-Agent': 'NeuroLabPA-img-lab (https://github.com/michalstankiewicz4-cell/NeuroLabPA)'}


def fetch(url, start, end):
    req = urllib.request.Request(url, headers={**UA, 'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def first_drawings(name, n):
    url = URL.format(name.replace(' ', '%20'))
    head = fetch(url, 0, 255)                      # nagłówek .npy: magia, wersja, długość opisu, opis (słownik)
    hlen = int.from_bytes(head[8:10], 'little') if head[6] == 1 else int.from_bytes(head[8:12], 'little')
    off = (10 if head[6] == 1 else 12) + hlen
    info = head[off - hlen:off].decode('latin1')
    assert "'descr': '|u1'" in info and '784' in info, info
    raw = fetch(url, off, off + n * 784 - 1)
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28, 28)


def to16(img28):
    im = Image.fromarray(img28).resize((SIZE, SIZE), Image.BOX)   # średnia z obszaru: cienkie kreski dają odcienie
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = np.clip(a * 1.6, 0, 1)                                    # pomniejszenie rozjaśnia kreski: wzmacniamy je
    return np.round(a * (LEVELS - 1)).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    pal = []
    for i in range(LEVELS):                                       # paleta: tło jasne (papier), kreska ciemna
        v = round(242 - (242 - 24) * i / (LEVELS - 1)); pal += [v, v, v]
    pal += [0] * (768 - len(pal))
    manifest = {'source': 'Quick, Draw! (Google), CC BY 4.0', 'size': SIZE, 'levels': LEVELS, 'cols': COLS, 'per': N, 'categories': []}
    for en, pl in CATS:
        d = first_drawings(en, N)
        tiles = np.stack([to16(x) for x in d])
        rows = (len(tiles) + COLS - 1) // COLS
        sheet = np.zeros((rows * SIZE, COLS * SIZE), dtype=np.uint8)
        for k, t in enumerate(tiles):
            r, c = divmod(k, COLS); sheet[r * SIZE:(r + 1) * SIZE, c * SIZE:(c + 1) * SIZE] = t
        im = Image.fromarray(sheet, mode='P'); im.putpalette(pal)
        fn = en.replace(' ', '_') + '.png'
        im.save(os.path.join(OUT, fn), optimize=True)
        manifest['categories'].append({'file': fn, 'name': pl, 'quickdraw': en, 'count': int(len(tiles))})
        print(f'{pl:10s} {len(tiles)} rysunków → {fn}', flush=True)
    with open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.exit(main())
