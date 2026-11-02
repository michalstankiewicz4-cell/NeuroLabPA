# Konwencje

## Wersje i historia
- Jedno źródło numeru wersji: `js/version.js` (`APP_VERSION`). Zmieniany razem z wpisem w `CHANGELOG.md`.
- Schemat `0.MINOR.PATCH`: MINOR przy nowej funkcji, PATCH przy poprawce lub zmianie ustawień. W SDF Walkers zmiana listy wejść sieci zawsze podnosi MINOR, bo unieważnia wyeksportowane modele.
- `ROADMAP.md`: plan, `CHANGELOG.md`: co się zmieniło, `notatki/`: reszta.

## Commity i GitHub
- Repo: `michalstankiewicz4-cell/NeuroLabPA`, gałąź `main`, publiczne. Strona: GitHub Pages (`.github/workflows/static.yml`), odświeża się 1–2 min po pushu.
- Commity robione przez Claude'a: committer `Claude <noreply@anthropic.com>`, autor `michalstankiewicz4-cell <248485489+michalstankiewicz4-cell@users.noreply.github.com>`. Prywatny adres e-mail nie trafia do publicznej historii.
- Tagi wersji: pominięte, bo środowisko Claude'a nie może ich wysyłać.
- Lokalnie trzymamy tylko pliki z repo. `dist/` jest ignorowany i usuwany po budowaniu.

## Uruchamianie
- Laboratoria (`lang/`, `img/`) używają Web Workerów i `fetch`, więc **nie działają z pliku otwartego z dysku**. Trzeba je uruchomić przez serwer (`python3 -m http.server` w katalogu repo, potem `http://localhost:8000`) albo przez GitHub Pages.
- Skrypty stron to zwykłe `<script>` bez modułów ES. Kolejność wczytywania w HTML ma znaczenie.
- Silnik jest wspólny: `lang/js/engine.js` (CPU + autograd), `lang/js/wasm.js` (WASM SIMD), `lang/js/gpu.js` (WebGPU). `img/js/worker.js` wczytuje je z `../../lang/js/`.

## Budowanie
- Wersja SDF Walkers jako jeden plik na claude.ai: `python3 tools/build_artifact.py` → `dist/sdf-walkers.html`.
- Moduł WASM (mnożenie macierzy i uwaga): źródło `lang/tools/matmul.wat`. Kompilacja: `npm install wabt@1.0.36` w dowolnym katalogu, potem `NODE_PATH=…/node_modules node lang/tools/build_wasm.js` → `lang/js/wasm.js` (base64, nie edytować ręcznie).
- Dane laboratorium obrazów: `python3 img/tools/build_quickdraw.py` (numpy, pillow; potrzebny dostęp do `storage.googleapis.com`). Pobiera tylko pierwsze 2000 rysunków każdej kategorii przez zapytania o fragment pliku.
- Dane laboratorium języka: pliki z Wolnych Lektur i Wikipedii są już w `lang/data/`. Każdy zestaw ma `LICENCJA.txt`, a lista jest w `manifest.json`.

## Testy
- Gradienty każdej nowej operacji silnika sprawdzamy numerycznie (różnice skończone) w Node przed użyciem.
- Nowe ścieżki obliczeń porównujemy ze starymi: WASM kontra JS (wyniki zgodne do ~1e-7), KV cache kontra pełne przeliczenie (~1e-5), GPU kontra CPU (test zgodności wbudowany w strony).
- Strony testujemy w Chromium bez okna (Playwright): klik „Ucz”, sprawdzenie metryk, błędów w konsoli i zrzut ekranu.
- WebGPU w środowisku bez karty: `chromium.launch(channel='chromium', args=['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader'])`. To emulacja na procesorze: sprawdza poprawność, nie szybkość.
- Serwer testowy uruchamiamy w tym samym poleceniu co test, bo procesy w tle znikają.
