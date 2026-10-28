// Laboratorium języka · dane: przykładowy tekst i tokenizacja po literach.

// Oryginalny tekst napisany na potrzeby laboratorium (nie cytat). Celowo krótki: wystarcza,
// żeby strona działała od razu, ale do sensownej nauki trzeba wgrać więcej tekstu.
const SAMPLE_TEXT = `Na końcu ulicy stał dom, w którym nikt nie mieszkał od lat. Okna były zabite deskami, a drzwi otwierały się same, kiedy wiał wiatr od rzeki. Dzieci omijały go szerokim łukiem, choć żadne nie umiało powiedzieć dlaczego.
Stary Zygmunt mówił, że w piwnicy ktoś wciąż liczy. Po cichu, równo, jak zegar, który zapomniał, że się zepsuł. Raz, dwa, trzy. Raz, dwa, trzy. Nikt mu nie wierzył, bo Zygmunt mówił wiele rzeczy.
Pewnej nocy w oknie na piętrze zapaliło się światło. Słabe, żółte, jakby ktoś trzymał świecę i bał się podejść bliżej. Sąsiadka spod siódemki widziała to dokładnie. Rano przysięgała, że światło mrugnęło do niej trzy razy.
Jesienią mgła przychodzi tu wcześnie. Kładzie się na trawie, potem na płotach, a w końcu wchodzi do sieni i czeka. Ludzie mówią, że mgła nie ma zapachu, ale to nieprawda. Pachnie mokrym kamieniem i czymś słodkim, czego nie da się nazwać.
Marta wróciła do miasta po dwudziestu latach. Pamiętała ulicę inaczej: szerszą, jaśniejszą, pełną głosów. Teraz słyszała tylko własne kroki i, gdzieś daleko, cichy stuk, jakby ktoś pukał w ścianę od środka.
Zatrzymała się przed domem. Drzwi były uchylone. Na progu leżał klucz, czysty i błyszczący, zupełnie jakby czekał właśnie na nią.
Weszła. W środku było ciepło. Na stole stała filiżanka herbaty, jeszcze parująca. Na krześle wisiał płaszcz, który znała z dzieciństwa, szary, z jednym guzikiem przyszytym czerwoną nicią.
Ktoś zszedł po schodach. Kroki były lekkie, nierówne, jakby osoba na górze uczyła się chodzić od nowa. Marta nie odwróciła się. Wiedziała, że jeśli się odwróci, zobaczy siebie.
W mieście mówiło się potem, że dom w końcu ktoś kupił. Okna odsłonięto, a w ogrodzie zakwitły białe róże. Tylko listonosz zauważył, że na skrzynce są dwa nazwiska, oba takie same.
Zima przyszła szybko. Śnieg przykrył ulicę, a dzieci znów zaczęły chodzić koło domu, bo z bliska słychać było muzykę. Ktoś grał na pianinie tę samą melodię, w kółko, bez błędu i bez końca.
Raz, dwa, trzy. Raz, dwa, trzy.
`;

// Słownik liter: każdy znak, który pojawia się w tekście, dostaje numer.
function buildVocab(text) {
  const chars = [...new Set(text)].sort((a, b) => a.localeCompare(b, 'pl'));
  const stoi = new Map(chars.map((c, i) => [c, i]));
  return { chars, stoi };
}
function encode(text, stoi) { const out = new Int32Array(text.length); for (let i = 0; i < text.length; i++) out[i] = stoi.get(text[i]) ?? 0; return out; }

// Ujednolicenie tekstu: jednakowe końce linii, bez podwójnych spacji i znaków sterujących.
function cleanText(t) {
  return t.replace(/\r\n?/g, '\n').replace(/[\t ]/g, ' ').replace(/[^\S\n]{2,}/g, ' ').replace(/\n{3,}/g, '\n\n').replace(/[\u0000-\u0008\u000b-\u001f]/g, '');
}
const POLISH_VOWELS = new Set([...'aąeęioóuyAĄEĘIOÓUY']);
