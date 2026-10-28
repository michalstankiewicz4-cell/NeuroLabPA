# Changelog

Wszystkie wersje powstały 28 października 2026, w jednej sesji pracy.
Wersje oznaczone ⚠ zmieniają listę wejść sieci, więc modele wyeksportowane wcześniej do nich nie pasują (oceny dalej działają).

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
