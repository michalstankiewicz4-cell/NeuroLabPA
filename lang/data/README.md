# Teksty przykładowe

Przycisk **Użyj przykładu** wczytuje wszystkie pliki wymienione w `manifest.json` i skleja je w jeden tekst do nauki.
Gdy lista jest pusta (albo strona jest otwarta z dysku), używa krótkiego tekstu z `js/data.js`.

Dodanie lektury: plik TXT z wolnelektury.pl (domena publiczna) wrzuć do tego folderu i dopisz do `manifest.json`:

```json
{ "files": [
  { "file": "latarnik.txt", "title": "Latarnik", "author": "Henryk Sienkiewicz" }
] }
```

Stopka licencyjna Wolnych Lektur (na końcu pliku, po linii z `-----`) jest pomijana przy wczytywaniu do nauki, ale w plikach zostaje. Źródło i licencja tekstów: `LICENCJA.txt`.
