# Decyzje i lekcje

## Silnik
- **Własny autograd w JS zamiast biblioteki**: żeby każde działanie dało się przeczytać i zrozumieć. Każda operacja ma ręcznie napisane przejście wstecz, sprawdzone numerycznie.
- **Trening w Web Workerze**: strona nigdy się nie zawiesza, a przycisk „Stop” działa w trakcie nauki.
- **WASM SIMD tylko dla najcięższych części** (mnożenie macierzy, iloczyny w uwadze). Softmax i jego pochodna zostają w JS, bo WASM nie ma funkcji wykładniczej, a ta część jest tania. Kod WASM jest napisany ręcznie w `.wat`.
- **GPU: cały krok nauki nagrany raz i odtwarzany.** Przesyłanie danych między procesorem a kartą przy każdym działaniu zjadłoby zysk. Do procesora wraca tylko błąd, a wagi co kilka sekund. Wymiar głowy uwagi musi być podzielny przez 4 (WASM).
- **KV cache przy rysowaniu**: klucze i wartości narysowanych pikseli trzymamy w pamięci, więc każdy piksel liczymy raz. 15× szybciej, ten sam wynik.
- **Najlepsza wersja sieci (wczesne zatrzymanie)**: wagi z momentu najniższego błędu sprawdzającego. Wracamy do nich tylko przy wzroście błędu o więcej niż 0,01 (mniejsze różnice to szum pomiaru, a wcześniej komunikat pojawiał się bez powodu).

## Laboratorium języka
- **Litery zamiast tokenów na start**: przy małych danych każda litera pojawia się tysiące razy, warstwa wyjściowa jest mała, a widać, jak sieć sama odkrywa polską odmianę. Tokeny BPE doszły później jako osobny szczebel.
- **BPE jak w GPT-2**: najpierw podział na słowa (spacja należy do słowa po niej), sklejenia tylko w obrębie słowa, liczenie par na rodzajach słów. Nauka trwa ok. 1 s zamiast minut.
- **Błąd na literę** przy tokenach (błąd na token / znaki na token), inaczej porównanie liter z tokenami byłoby nieuczciwe.
- **Przy tokenach końcowa spacja początku tekstu jest odcinana**, bo w nauce spacja należy do następnego słowa (problem „token healing”).
- **Sprawdzanie RNN czyta tekst ciągiem** z rozgrzewką 40 liter, a sprawdzanie transformera liczy tylko drugą połowę okna. Dzięki temu każda zgadywana litera ma przed sobą kontekst.
- **Dostrajanie do rozmowy z maskowaniem**: błąd liczony tylko na kwestiach Sieci, jak przy prawdziwych asystentach.
- **Dialogi napisał Claude** (CC0), bo nie było otwartego polskiego zbioru rozmów. Sieć mówi skromnie i nie podaje faktów o świecie.

## Laboratorium obrazów
- **Dane jako pliki PNG** (siatka rysunków): widać w folderze dokładnie to, z czego uczy się sieć.
- **Kategoria jako numer**: sieć nie zna słów, uczy się tylko powiązania numeru z kształtami. Opis zdaniem wymagałby kodera tekstu i rysunków z opisami.
- **VAE, GAN i dyfuzja na warstwach gęstych** zamiast splotowych: przy 16×16 wystarczą i uczą się w minuty.
- **GAN bez sztuczek nie działał**: krytyk wygrywał, a generator rysował to samo dla każdej kategorii. Pomogły: szum na wejściu krytyka (malejący), etykieta 0,9, wolniejszy krytyk, Adam β1 = 0,5 i przede wszystkim **dyskryminator projekcyjny**.
- **Dyfuzja zgaduje czysty rysunek, nie szum**: z przewidywaniem szumu (oryginalny DDPM) mała sieć gęsta zostawiała zaszumione rysunki. Ta sama metoda w innej parametryzacji dała najlepsze rysunki ze wszystkich kroków.
- **Siła podpowiedzi (classifier-free guidance)**: 10% nauki bez kategorii, przy rysowaniu mieszanka odpowiedzi z kategorią i bez. Na tym samym mechanizmie stoi Wyobraźnia: mieszanie kategorii i obszary.

## SDF Walkers
- **Strona zawieszała się na Windows** (bez komunikatu w konsoli). Kompilator shaderów DirectX (FXC przez ANGLE) rozwijał pętle i wklejał funkcję kształtu w każde wywołanie. Pomogły: granice pętli nieznane kompilatorowi (`uZero`), normalne w jednej pętli (czworościan), kompilacja w tle (`KHR_parallel_shader_compile`), renderowanie po kawałku z czekaniem na kartę (fence).
- **Reguła kontrastu kolorów (Oklab)** usunęła zlewające się zestawy, ale zawęziła paletę koszul.
- **Bad bez wskazania części uczy tylko wyjścia `good`**: świadoma decyzja, wariant MIL / noisy-OR odrzucony.
