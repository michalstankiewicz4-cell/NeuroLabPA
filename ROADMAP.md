# Roadmap

Plany z rozmów projektowych, w kolejności, w jakiej najpewniej je zrobimy.

## W toku
- **Partia ocen od Claude'a.** Claude ocenia zombie (te same miniatury co Ty: przód i bok, pierwsza klatka chodu) i zapisuje oceny jako `rater: "claude"`. Strona odbiera je na żywo i uczy się z nich z mniejszą wagą niż z Twoich (`CLAUDE_WEIGHT`).

## Laboratorium języka
- **Krok 4: transformer.** Uwaga (attention): każda litera sama wybiera, na które wcześniejsze patrzeć. Mapa uwagi na żywo.
- **Przełącznik CPU/GPU (WebGPU)**, razem z krokiem 4. Silnik w shaderach WGSL, w którym cały krok nauki zostaje na karcie graficznej. Porównanie szybkości CPU i GPU dla RNN (czyta po kolei, więc zyskuje mało) i dla transformera (wszystkie pozycje naraz, więc zyskuje dużo). To w praktyce pokazuje, dlaczego transformer wygrał.
- Tokeny większe niż litera (BPE), potem rozmowa: dostrajanie na dialogach.

## Następne
- **Różnorodność kolorów ubrań.** Reguła kontrastu (0.16.0) usunęła zlewające się zestawy, ale przy okazji zawęziła paletę: koszule rdzawe i fioletowe to razem 52% (zamiast 33%), a zielone i musztardowe spadły do 8–9%. Pomysł: każdy kolor w wariancie jasnym i ciemnym, wtedy reguła ma z czego wybierać, zamiast odrzucać całe kolory.
- **Zakładka Review.** Przeglądasz oceny Claude'a: „Agree” albo poprawka werdyktu i części. Wagi: niesprawdzona ocena Claude'a < potwierdzona < Twoja poprawka. Statystyka zgodności między Wami, ogólna i dla każdej części ciała.
- **Widok z tyłu na kartach ocen.** Włosy i ubrania najlepiej widać od tyłu, a tej strony nikt teraz nie ocenia.

## Później
- **Styl ruchu jako cecha okazu**: rodzaj chodu (szuranie, kuśtykanie, chwiejny krok), tempo, zamach rąk, sposób ataku i upadku. Dopiero wtedy sens ma **druga sieć do animacji**, z kartami pokazującymi ruch zamiast pojedynczej klatki. Dopóki ruch jest wspólny dla wszystkich zombie, uwagi o animacji poprawia się w kodzie.
- **Porównanie architektur sieci** na tych samych ocenach (ta sama miara: zgodność w walidacji krzyżowej):
  - 0, 1 albo 2 warstwy ukryte,
  - sieć modułowa (osobne podsieci: głowa, ciało, kolory), w której wyjście części widzi tylko swoje wejścia,
  - kilka sieci głosujących (ensemble) plus niepewność, a do oceny pokazywane zombie, przy których sieci się kłócą.
  Ma sens od ok. 50+ ocen.
- **Loader dla silnika VS**: wczytanie arkusza i JSON-a, billboard w świecie 3D, wybór kierunku z ruchu.
- **Generator nastawiony na krytyka**: zamiast losować i filtrować, uczyć generator parametrów tak, by podobał się sieci (pomysł GAN-owy). Ma sens po kilkuset ocenach.

## Pomysły odłożone
- Tag „Overall look” dla Bad, którego nie da się przypisać do części.
- Uczenie części z ocen Bad bez tagów (multiple-instance / noisy-OR): sieć sama obwinia najbardziej podejrzaną część. Odłożone świadomie: na razie Bad bez tagów uczy tylko wyjście `good`.
