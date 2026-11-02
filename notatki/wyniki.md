# Wyniki

Wszystkie pomiary z przeglądarki (Chromium bez okna) albo z Node na tym samym silniku, procesor w środowisku Claude'a. Na komputerze użytkownika czasy mogą być inne.

## Laboratorium języka (Grabiński, 43 nowele, 1,14 mln znaków)

Błąd sprawdzający na literę (mniej = lepiej). Przy tokenach błąd na token jest dzielony przez średnią liczbę znaków na token.

| Konstrukcja | Litery | BPE 512 |
|---|---|---|
| Bigram | 2,52 | 1,96 |
| Sieć z oknem (okno 8, emb 16, 128 neuronów, 3000 kroków) | 2,17–2,22 | 2,18 |
| Zwykła RNN (pamięć 64, 2000 kroków) | 2,15 | – |
| GRU (pamięć 64, 1000 / 2000 kroków) | 2,19 / 2,08 | 2,09 (1000) |
| Transformer 2×32, 4 głowy, okno 64 (1000 / 2000 kroków) | 2,22 / 2,05 | 2,04 / **2,00** |

- Wikipedia (42 artykuły, 1,16 mln znaków, 262 różne znaki): bigram 2,62, sieć z oknem 2,28. Wikipedia jest trudniejsza: nazwy, liczby, obce alfabety.
- Mały przykład (2 tys. znaków): sieć z oknem uczy się na pamięć (błąd treningowy 0,015, sprawdzający 4,4). Dobra demonstracja przeuczenia.
- Głowy transformera po 2000 krokach: warstwa 1 patrzy 1–2 litery wstecz (jedna w 85% na poprzednią literę), warstwa 2 ok. 12 liter wstecz.
- Zasięg pamięci: po 30 literach do zwykłej RNN dociera ok. 10⁻⁶ sygnału, do GRU ok. 1%.
- Rozmowa (krok 5, 140 dialogów): po dostrojeniu odpowiedzi mają formę rozmowy (krótkie, we własnej kolejce, kropka na końcu), treść jest bez sensu. Wczesne zatrzymanie po ok. 170 krokach, bo sieć zapamiętuje dialogi.

## Laboratorium obrazów (Quick, Draw!, 12 kategorii × 2000, 16×16, 16 odcieni)

| Konstrukcja | Ustawienia | Wynik | Czas (CPU, WASM) |
|---|---|---|---|
| Statystyka pikseli | – | 1,74 bita/piksel | < 1 s |
| Piksel po pikselu | 2×48, 4 głowy | 1000 kroków: 1,47 · 3000: 1,26 · 6000: 1,16 bita/piksel | ok. 0,25–0,3 s/krok |
| VAE | z = 8, 256 neuronów, 4000 kroków | błąd 0,35 na piksel; rozmyte, kategorie czytelne | ok. 40 s |
| GAN | 256 neuronów, 6000 kroków | krytyk 63% / 26%; ziarniste, kategorie się różnią | ok. 3 min |
| Dyfuzja | T = 100, 3×512, 4000 kroków | błąd zgadywania 0,21; **najlepsze rysunki** | ok. 4 min |

- Piksel po pikselu, co rysuje: po 1000 krokach kreski w poziomie, po 3000 pierwsze kształty (drzewo, gwiazda), po 6000 dom, ryba, księżyc. Zob. `docs/img/img-lab-postep.png`.
- Porównanie VAE / GAN / dyfuzji: `docs/img/img-lab-vae-gan-dyfuzja.png`. Wyobraźnia (drzewo z głową kota): `docs/img/img-lab-wyobraznia.png`.

## Szybkość silnika

| Co | JS | WASM SIMD |
|---|---|---|
| Mnożenie macierzy (mld mnożeń/s) | 0,38 | 3,45 |
| Krok: sieć z oknem / GRU / transformer (język) | 15 / 93 / ok. 375 ms | 5 / 32 / 109 ms |
| Krok transformera obrazów (przed / po uwadze w WASM) | 626 ms | 315 ms |
| Rysowanie obrazka piksel po pikselu (bez / z KV cache) | 3,4 s | 0,23 s |

- GPU (WebGPU) sprawdzone tylko w emulacji: wyniki zgodne z CPU (błąd do 5–7 cyfr, wagi ~1e-4), szybkość nieznana.
