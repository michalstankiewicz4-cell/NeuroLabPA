# Teksty przykładowe

Lista **Przykład** na stronie laboratorium pokazuje zestawy z `manifest.json`. Przycisk **Wczytaj przykład** skleja pliki zestawu w jeden tekst do nauki.
Gdy zestawów brak (albo strona jest otwarta z dysku), używa krótkiego tekstu z `js/data.js`.

| Folder | Zawartość | Licencja |
|---|---|---|
| `grabinski/` | 43 nowele i opowiadania Stefana Grabińskiego z Wolnych Lektur, ok. 1,1 mln znaków | domena publiczna, szczegóły w `grabinski/LICENCJA.txt` |
| `rozmowy/` | 140 krótkich dialogów „Ty: … / Sieć: …” do dostrajania w kroku 5 (nie jest przykładem do wyboru) | CC0, napisane przez Claude'a, `rozmowy/LICENCJA.txt` |
| `wikipedia/` | 42 „Artykuły na medal” z polskiej Wikipedii z różnych dziedzin, ok. 1,2 mln znaków | CC BY-SA 4.0, autorzy, wersje i zmiany w `wikipedia/LICENCJA.txt` |

Oba zestawy mają podobną wielkość, żeby porównanie tej samej sieci na prozie i na tekście encyklopedycznym było uczciwe.

Nowy zestaw: folder z plikami TXT i `LICENCJA.txt` plus wpis w `manifest.json`:

```json
{ "id": "moj", "name": "Mój zestaw", "label": "Mój", "dir": "moj/", "license": "moj/LICENCJA.txt",
  "note": "Opis źródła.", "strip": null, "files": [ { "file": "tekst.txt", "title": "Tytuł" } ] }
```

`"strip": "wolnelektury"` pomija przy wczytywaniu notę licencyjną Wolnych Lektur (po linii `-----`) i linię z ISBN; w plikach zostają.
