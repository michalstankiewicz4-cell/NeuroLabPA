# Changelog

Wszystkie wersje powstały 28 października 2026, w jednej sesji pracy.
Wersje oznaczone ⚠ zmieniają listę wejść sieci, więc modele wyeksportowane wcześniej do nich nie pasują (oceny dalej działają).

## 0.21.0
- Laboratorium języka, **krok 4: transformer** w wersji „tylko dekoder”, jak GPT.
  - Uwaga przyczynowa z kilkoma głowami, wektory pozycji, normalizacja warstwy, GELU i połączenia omijające.
  - Do wyboru: warstwy, szerokość, głowy i okno.
- **Mapa uwagi:** klikasz literę i widzisz, na które wcześniejsze patrzy wybrana głowa. Przy każdej głowie jest podane, jak daleko średnio sięga.
  - Po 2000 krokach na Grabińskim głowy pierwszej warstwy patrzą 1–2 litery wstecz (jedna w 85% na poprzednią literę), a głowy drugiej warstwy ok. 12 liter wstecz.
- **Przełącznik CPU/GPU** dla transformera.
  - Nowy silnik `lang/js/gpu.js` w WebGPU: 18 shaderów WGSL dla każdego działania w przód i wstecz, przycięcia gradientu i Adama.
  - Cały krok nauki jest nagrany raz i odtwarzany na karcie; do procesora wraca tylko błąd, a wagi co kilka sekund (do próbek i map).
- **Test zgodności przy pierwszym wyborze GPU:** ten sam start i ta sama porcja tekstu na CPU i GPU. Błąd zgadza się do 5–7 cyfr, wagi po kroku różnią się o ok. 1e-4 (zaokrąglenia). Bez WebGPU opcja jest wyłączona z wyjaśnieniem.
- Czas kroku nauki (ms, CPU albo GPU z nazwą karty) w metrykach i w nowej kolumnie tabeli porównania.
- Silnik CPU: `addPos`, `layerNorm`, `gelu`, `causalAttention` z ręcznie liczonym gradientem, sprawdzone numerycznie.
- Wyniki na Grabińskim (błąd sprawdzający):
  - transformer 2×32, 4 głowy, okno 64, 2000 kroków na CPU: 2,05;
  - GRU: 2,08;
  - zwykła RNN: 2,15;
  - sieć z oknem: 2,22;
  - bigram: 2,52.
- Silnik GPU sprawdzony tu tylko w emulacji (SwiftShader): wyniki są poprawne, ale prawdziwą szybkość widać dopiero na prawdziwej karcie.

## 0.20.0
- Laboratorium języka, **krok 3: sieć rekurencyjna** do wyboru w dwóch wersjach: zwykła RNN (Elman 1990) i GRU z bramkami (2014).
  - Nauka na kawałkach tekstu z obciętą propagacją wstecz i przycinaniem gradientu.
  - Sprawdzanie czyta tekst ciągiem: pierwsze 40 liter rozgrzewa pamięć.
- Nowe wizualizacje kroku 3:
  - **Neurony pamięci:** stan wybranego neuronu pod każdą literą tekstu. Strona sama znajduje neurony, które pokrywają się z cechami tekstu: litera czy odstęp, pozycja w słowie, samogłoska, cudzysłów, nawias.
  - **Zasięg pamięci:** jaka część sygnału uczącego dociera do pamięci sprzed 1–59 liter, w skali logarytmicznej, z porównaniem zakończonych nauk. W zwykłej RNN sygnał zanika wykładniczo, w GRU zostaje.
  - Mapa liter jak w kroku 2.
- Silnik:
  - nowe operacje: dodawanie, odejmowanie i mnożenie po elementach, sigmoida, wycinanie i sklejanie wierszy i kolumn;
  - komórki `rnnCell` i `gruCell`;
  - obcinanie gradientu w Adam;
  - gradienty sprawdzone numerycznie.
- **Najlepsza wersja sieci (wczesne zatrzymanie)** dla kroków 2 i 3. Strona zapamiętuje wagi z momentu najniższego błędu sprawdzającego. Gdy pod koniec błąd wyraźnie wzrośnie, wraca do nich i to zgłasza. Najlepszy punkt jest zaznaczony na wykresie.
- **Zapisz model / Wczytaj model** (plik JSON z wagami, słownikiem liter i nazwą tekstu). Wczytany model z tym samym słownikiem jest sprawdzany na bieżącym tekście i trafia do tabeli. Model z innym słownikiem działa tylko do generowania.
- Wynik na Grabińskim (błąd sprawdzający):
  - GRU (pamięć 64, 2000 kroków): 2,08;
  - zwykła RNN: 2,15;
  - sieć z oknem (3000 kroków): 2,22;
  - bigram: 2,52.
  GRU przy tym ustawieniu uczy się ok. 5 minut.

## 0.19.0
- Laboratorium języka: lista **Przykład** z trzema tekstami do wyboru:
  - „Proza grozy: Stefan Grabiński”: 43 nowele, ok. 1,1 mln znaków, domena publiczna;
  - „Encyklopedia: polska Wikipedia”: 42 „Artykuły na medal” z różnych dziedzin, ok. 1,16 mln znaków, CC BY-SA 4.0;
  - krótki tekst Claude'a do pokazania uczenia na pamięć.
- Każdy zestaw ma swój folder i `LICENCJA.txt`. Przy Wikipedii są tam: autorzy (link do historii edycji), dokładna wersja każdego artykułu i opis zmian (zostały tylko akapity i nagłówki).
- Pierwsze porównanie, sieć z oknem (okno 8, embedding 16, 128 neuronów) kontra bigram:
  - Grabiński: błąd sprawdzający 2,12 wobec 2,52;
  - Wikipedia: 2,28 wobec 2,62.
  Wikipedia ma 262 różne znaki (grecki, cyrylica, symbole), Grabiński 114, więc warstwa wyjściowa jest większa i nauka wolniejsza.

## 0.18.2
- Laboratorium języka: przykład to teraz 43 nowele i opowiadania Stefana Grabińskiego z Wolnych Lektur (domena publiczna, ok. 1,1 mln znaków). Źródło, licencja i lista utworów: `lang/data/LICENCJA.txt`; oryginalne noty licencyjne zostają w plikach.
- Na tym tekście sieć z oknem wyraźnie wygrywa z bigramem (błąd sprawdzający 2,12 wobec 2,52; waha się między 8 literami zamiast 12) i nie uczy się na pamięć.
- Przy wczytywaniu do nauki pomijana jest też linia z numerem ISBN.

## 0.18.1
- Laboratorium języka: **Użyj przykładu** wczytuje lektury z folderu `lang/data/` (lista w `manifest.json`), a nota licencyjna Wolnych Lektur na końcu pliku jest pomijana. Bez lektur, albo z pliku otwartego z dysku, zostaje krótki tekst wbudowany.

## 0.18.0
- Nowa strona **Laboratorium języka** (`lang/`): własna sieć uczy się polskiego od zera, litera po literze. Link w nagłówku strony głównej.
- Własny silnik uczenia w czystym JS (`lang/js/engine.js`): tensory, automatyczne liczenie gradientów (autograd), Adam, PCA. Sprawdzony testem gradientów numerycznych.
- Trening w osobnym wątku (Web Worker), więc strona nie zawiesza się w trakcie nauki.
- Krok 1: **bigram** (liczenie par liter) z tabelą „po jakiej literze jaka”.
- Krok 2: **sieć z oknem** (Bengio 2003): embeddingi liter, warstwa tanh, wybór okna, rozmiaru embeddingu i liczby neuronów. Mapa liter (PCA embeddingów), na której widać, jak sieć sama grupuje litery.
- Wczytywanie własnych plików .txt (kilka naraz), krótki przykładowy tekst na start, podział 90/10 na tekst do nauki i do sprawdzania.
- Na żywo: wykres błędu (treningowy i sprawdzający, poziom „bez wiedzy”), bity na znak, „waha się między N literami”, próbki tekstu z temperaturą, ostrzeżenie, gdy sieć uczy się na pamięć.
- Tabela porównania wszystkich nauk na tym samym tekście.

## 0.17.0
- Folder `data/` w repo: `claude-ratings.json` (160 ocen Claude'a, w tym 11 oznaczonych jako nieaktualne) i `model.json` (sieć wytrenowana na 149 aktualnych ocenach Claude'a).
- Kopia strony poza claude.ai (GitHub Pages, serwer lokalny) sama dołącza oceny Claude'a z `data/` do nauki. Nie są zapisywane w przeglądarce, więc eksport i pamięć lokalna zawierają tylko Twoje oceny.
- Przycisk **Load repo model** wczytuje `data/model.json` jednym kliknięciem. (Przeglądarka nie pozwala stronie ustawić domyślnego folderu w oknie wyboru pliku, stąd osobny przycisk.)
- Eksportowany model zapisuje, czyje oceny go wytrenowały (`rater`).

## 0.16.2
- Loader nad podglądem: „Compiling shader” (z czasem i wyjaśnieniem, gdy trwa dłużej niż 3 s), potem „Rendering sprite sheet” z paskiem i liczbą klatek. Przy szybkich przeliczeniach po edycji się nie pojawia.
- Postęp jest prawdziwy: renderer czeka na faktyczne zakończenie pracy karty graficznej po każdej animacji (fence), zamiast tylko kolejkować rysowanie. Wcześniej cała praca wykonywała się na końcu, w jednym blokującym kroku.
- Miniatury w siatce ocen pulsują, dopóki się nie wyrenderują.

## 0.16.1
- Naprawa: strona zawieszała się przy otwieraniu (bez komunikatu w konsoli). Na Windowsie przeglądarki tłumaczą WebGL na DirectX, którego kompilator rozwija pętle i wkleja funkcję kształtu w każde miejsce wywołania. Shader urósł na tyle, że kompilacja blokowała kartę. Pętle mają teraz granicę nieznaną kompilatorowi, a normalne są liczone w jednej pętli zamiast 6 wywołań.
- Shadery kompilują się w tle (`KHR_parallel_shader_compile`), a w podglądzie widać „COMPILING SHADER…”. Arkusz renderuje się po jednej animacji na klatkę obrazu, więc strona odpowiada w trakcie.
- W konsoli pojawia się informacja, ile trwała kompilacja.

## 0.16.0 ⚠
Poprawki po pierwszej partii 96 ocen Claude'a, w której połowa ocen Bad wskazywała błędy wspólne dla wszystkich zombie, a nie cechy okazów:
- Ręce nie rzucają już cienia na klatkę piersiową. Kwantyzacja zamieniała ten cień w ukośny pas wyglądający jak szarfa.
- Łachmany: postrzępiony dół i kilka dużych dziur w innym miejscu u każdego zombie, zamiast gęstego, regularnego wzoru, który wyglądał jak kamuflaż.
- Reguła palety porównuje kolory w przestrzeni Oklab (tak jak widzi oko), więc odrzuca też szare na szarym i ciemne na ciemnym, a nie tylko podobne odcienie. Ubrania mają różne jasności.
- Wejścia sieci: zamiast 2 odległości odcieni są 3 kontrasty Oklab (koszula↔spodnie, koszula↔skóra, spodnie↔skóra). Razem 56 wejść.
- Oceny oznaczone `stale` (opisujące naprawiony błąd renderowania) są pomijane przy nauce. Tak oznaczono 11 ocen Claude'a.

## 0.15.0
- Oceny odbierane na żywo: strona subskrybuje bazę, więc oceny zapisane gdzie indziej (np. partia od Claude'a) od razu trenują sieć.
- Każda ocena wie, kto ją wystawił (`rater: you | claude`). Przełącznik „Learn from: both / my ratings / Claude's”. Oceny Claude'a mają przy „both” wagę 0,6.
- `ROADMAP.md` z planami. `tools/build_artifact.py` składa projekt w jeden plik dla claude.ai.

## 0.14.0
- Podział projektu na pliki i foldery (`index.html`, `css/`, `js/`), README, CHANGELOG, numer wersji w stopce.
- Eksport poza claude.ai działa jako zwykłe pobieranie pliku.

## 0.13.0
- **Export ratings**, **Export model**, **Import model**. Plik modelu zawiera listę wejść i jest odrzucany przy niezgodnej wersji.
- Stare oceny (sprzed nowych cech) są czytane jako „łysy mężczyzna w koszuli i ciemnych butach”, czyli dokładnie to, co wtedy było oceniane.

## 0.12.0 ⚠
- Obuwie: boso, trampki, buty robocze, półbuty, kalosze. Wysokie buty mają cholewkę na łydce.

## 0.11.0 ⚠
- Płeć żeńska: węższe barki i talia, szersze biodra, częściej długie włosy.
- Nowe stroje: łachmany, bez koszulki / krótki top, sukienka, szorty.
- Zasada bez nagości jest wpisana w shader: klatka piersiowa kobiet jest zawsze zakryta, niezależnie od stroju i edytora.

## 0.10.0 ⚠
- Wzrost, budowa, barki, włosy (łysy, krótkie, długie, wyłysiałe) w 4 kolorach, rodzaj ubrania (koszula, kurtka, koszula szpitalna, bez rękawów, kombinezon).
- Pogrubione pasma długich włosów i większe łyse placki, bo przy głowie ~8 px drobniejsze detale znikały.

## 0.9.0 ⚠
- Tagi to części ciała: Head, Torso, Arms, Legs, Skin colour, Clothes colour. Tag zawsze oznacza Bad, a Good czyści tagi.
- Sieć z 7 wyjściami (Good + 6 części) i maskowaniem nieznanych części.
- Network view: przesuwanie, zmiana rozmiaru (także z klawiatury), czytelne podpisy wejść.

## 0.8.0
- Network view: półprzezroczysty panel z siecią na żywo, krzywą błędu w trakcie nauki i „myśleniem” przy Smart roll.

## 0.7.0
- Ręce zgięte w łokciu i asymetryczne, szersze biodra, słabszy dithering (po recenzji partii 16 zombie).
- Reguły palety: koszula ≠ spodnie, koszula ≠ skóra, oczy kontrastowe względem skóry.
- Siatka oceniania i sieć neuronowa w przeglądarce. Oceny zapisują się w bazie artefaktu (tylko właściciel).

## 0.6.0
- Realistyczne proporcje człowieka (głowa ≈ 1/7,5 wzrostu).
- Edytor wszystkich cech okazu.
- Pixel-art polish: przesadzony chód z przodu, grubsze nogi, czyszczenie samotnych pikseli, dithering, selektywny kontur.

## 0.5.0
- Animacje idle, attack, hit, death. JSON arkusza dla silnika gry.

## 0.4.0
- Naprawa: przy całej czaszce postać znikała (utrata precyzji przy odejmowaniu kształtu „w nieskończoności”).
- Interfejs po angielsku.

## 0.3.0
- SDF Walkers: cała postać, cykl kroku, sprite sheet 8 klatek × 4 kierunki z przezroczystym tłem.

## 0.2.0
- SDF Morgue (`legacy/sdf-morgue.html`): głowa zombie w shaderze WebGL2, losowanie okazów z ziarnem.

## 0.1.0
- Prototyp w Pythonie (`tools/prototype/`): głowa zombie z SDF, raymarching w numpy, pixel art. Iteracje v1 → v5 z oceną każdego renderu.
