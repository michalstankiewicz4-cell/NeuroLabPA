# Sprawy otwarte i znane ograniczenia

## Do sprawdzenia
- **Prawdziwa szybkość GPU.** Silnik WebGPU był testowany tylko w emulacji (SwiftShader): wyniki są poprawne, ale nie wiemy, o ile karta przyspiesza naukę. Do zmierzenia w Chrome/Edge na Windows: transformer w obu laboratoriach, porównanie czasu kroku CPU kontra GPU w tabeli.
- **Obszary w Wyobraźni** mocno zależą od losowego szumu: część prób daje czytelną hybrydę, część plamy. Możliwe usprawnienia: domyślnie większa siła podpowiedzi dla obszarów, dłuższa nauka dyfuzji, kilka wersji naraz do wyboru.

## SDF Walkers (zombie)
- **Oceny Claude'a są mało przydatne dla sieci**: w drugiej partii 63 z 64 dostały „Good”, więc niewiele uczą. Liczą się oceny użytkownika.
- **11 ocen jest nieaktualnych** (sprzed poprawek cieni i podartych ubrań). Są oznaczone i nie biorą udziału w nauce.
- **Kolory koszul**: po regule kontrastu dominują rdza i fiolet (razem ok. 52%). Pomysł w ROADMAP: jasny i ciemny wariant każdego koloru.

## Laboratorium języka
- **„Jestem Michał”**: żeby sieć przedstawiała się imieniem, potrzebne są własne rozmowy w kroku 5 z wieloma wariantami pytań o imię. Trzeba usunąć sprzeczne przykłady (obecny zbiór mówi „Nie mam imienia”). Funkcja wgrywania własnych rozmów jest w ROADMAP.
- **Sieć z oknem na tokenach się nie poprawia**: przy ponad 600 tokenach ma za mało kroków na dużą warstwę wyjściową.
- **Rozmowa w małej skali to sama forma**: sieć nie rozumie pytań. Większa poprawa wymaga większego modelu (GPU), dłuższej nauki i większego, różnorodnego zbioru rozmów.
- **API Wikipedii odrzucało zapytania** z tego środowiska (limit). Artykuły pobrano jako zwykłe strony, po jednym co 2 s.

## Laboratorium obrazów
- **Piksel po pikselu potrzebuje długiej nauki** (rozpoznawalne kształty od ok. 3000–5000 kroków, na CPU 12–25 min).
- **GAN po 6000 krokach jest ziarnisty**: potrzebuje znacznie dłuższej nauki.
- **Dane są czarno-białe** (Quick, Draw! to rysunki kreską). Kolory dojdą z innym zbiorem, np. zombie z SDF Walkers.
