// Laboratorium obrazów · strona: dane, graf modelu, nauka, rysowanie, uwaga, porównanie.
const $ = id => document.getElementById(id);
const fmt = (v, d = 2) => v == null || !isFinite(v) ? '–' : v.toFixed(d).replace('.', ',');
const LN2 = Math.log(2), S = 16, N = 256, LV = 16;
const shade = v => Math.round(242 - (242 - 24) * v / (LV - 1));   // odcień 0..15 → szarość (papier → kreska)
const worker = new Worker('js/worker.js');

// ---------- kroki ----------
const STEPS = [
  { kind: 'stats', name: '1 · Statystyka pikseli', sub: 'każdy piksel osobno', ready: true,
    what: 'Dla każdej kategorii i każdego z 256 miejsc liczymy, jak często pojawia się każdy z 16 odcieni. Rysując, losujemy każdy piksel osobno z jego tabeli.',
    why: 'Odpowiednik bigramu z tekstu: najprostszy model, który w ogóle coś „wie” o rysunkach. Średni obrazek każdej kategorii już przypomina kota czy słońce. Jest punktem odniesienia dla wszystkich kolejnych konstrukcji.',
    lim: 'Piksele nic o sobie nie wiedzą. Kreska nie może się ciągnąć, bo każdy piksel jest losowany niezależnie, więc wychodzi szum w kształcie kategorii. Nie umie też narysować dwóch różnych kotów: każdy jest tym samym rozmytym średnim kotem z szumem.' },
  { kind: 'ar', name: '2 · Piksel po pikselu', sub: 'transformer, jak ImageGPT', ready: true,
    what: 'Obrazek czytamy jak tekst: wiersz po wierszu, 256 pikseli. Na początek wstawiamy numer kategorii. Transformer, ten sam co w laboratorium języka, na każdej pozycji patrzy na kategorię i wszystkie wcześniejsze piksele i zgaduje odcień następnego.',
    why: 'Usuwa wadę statystyki: każdy piksel zależy od już narysowanych, więc kreska może się ciągnąć, a oczy pasują do głowy. Tak działał ImageGPT (OpenAI, 2020). Uwaga sama odkrywa dwa wymiary: piksel bezpośrednio nad nim jest 16 pozycji wcześniej, a sieć uczy się tam patrzeć.',
    lim: 'Rysuje po kolei, więc wygenerowanie obrazka to 256 kroków. Przy większych obrazkach to bardzo wolne. Błąd na początku rysunku ciągnie się do końca, bo nie da się wrócić i poprawić. Dlatego dziś do obrazków używa się głównie dyfuzji.',
    more: 'Ile uczyć: po 1000 krokach sieć umie głównie ciągnąć kreskę w poziomie, od ok. 3000 kroków pojawiają się kształty (drzewo, gwiazda), po 5000–6000 rozpoznawalne domy, ryby i księżyce. Na procesorze 3000 kroków to ok. 12–15 minut, na karcie graficznej zwykle kilka razy krócej. Pozycje to 256 wyuczonych wektorów, po jednym na piksel. Sieć nie dostaje podpowiedzi, że obrazek ma 16 kolumn: musi to odkryć sama. Kategoria to po prostu jeszcze jeden „wyraz” w słowniku wejścia (16 odcieni + 12 kategorii).' },
  { kind: 'vae', name: '3 · Autoenkoder (VAE)', sub: 'ściśnij i odtwórz, 2013', ready: true,
    what: 'Koder ściska rysunek (razem z kategorią) do kilku liczb z, czyli punktu w „przestrzeni ukrytej”. Dekoder z samego z i kategorii odtwarza rysunek. Uczymy obu naraz: odtworzenie ma być jak najwierniejsze.',
    why: 'Nowy pomysł: zamiast rysować piksel po pikselu, sieć uczy się krótkiego opisu rysunku. Błąd ma drugi składnik (dywergencja KL), który zmusza punkty z do układania się jak zwykły rozkład normalny. Dzięki temu można wylosować dowolne z i dostać sensowny rysunek, a bliskie punkty dają podobne rysunki. Wszystkie piksele powstają naraz, w jednym przejściu.',
    lim: 'Rysunki są rozmyte. Gdy sieć nie wie, czy kreska ma być tu, czy obok, rysuje średnią z obu możliwości, czyli szarą smugę. Kilka liczb z nie mieści wszystkich szczegółów. To główna wada VAE i powód, dla którego powstały GAN-y.',
    more: 'Koder i dekoder to po jednej warstwie ukrytej (tanh). Sztuczka reparametryzacji: z = mu + rozrzut × losowy szum, dzięki czemu przez losowanie da się liczyć gradient. Przy 2 liczbach z można narysować całą przestrzeń ukrytą jako siatkę: zobacz „Przestrzeń ukryta” niżej.' },
  { kind: 'gan', name: '4 · GAN', sub: 'generator kontra krytyk, 2014', ready: true,
    what: 'Dwie sieci grają przeciwko sobie. Generator z losowego szumu (i kategorii) robi rysunek. Krytyk dostaje raz rysunek prawdziwy, raz podróbkę i ocenia, czy jest prawdziwy. Krytyk uczy się odróżniać, generator uczy się oszukiwać krytyka.',
    why: 'Usuwa rozmycie VAE: generator nie porównuje się piksel po pikselu z oryginałem, tylko musi zrobić coś, czego krytyk nie odróżni od prawdziwych rysunków. Szara smuga od razu zdradza podróbkę, więc generator uczy się ostrych kresek. Nikt nie mówi mu, jak ma wyglądać rysunek.',
    lim: 'Kapryśna nauka: gdy krytyk za szybko wygrywa, generator nie dostaje użytecznych wskazówek. Generator lubi też rysować w kółko to samo, ignorując kategorię i szum (załamanie trybów). Nie ma prostej miary postępu: błędy obu sieci to wynik gry, nie jakość rysunków. Potrzebuje długiej nauki.',
    more: 'Sztuczki stabilizujące: szum na wejściu krytyka (malejący w trakcie nauki), etykieta 0,9 zamiast 1 dla prawdziwych rysunków, wolniejszy krytyk, Adam z β1 = 0,5 i dyskryminator projekcyjny (Miyato 2018), w którym kategoria wpływa na ocenę bezpośrednio. Bez nich generator rysował to samo dla każdej kategorii. Szum generatora jest stały przez całą naukę: widać, jak zmienia się ten sam rysunek.' },
  { kind: 'diff', name: '5 · Dyfuzja', sub: 'z szumu do rysunku, 2020', ready: true,
    what: 'Nauka: do prawdziwego rysunku dodajemy losową ilość szumu (od odrobiny do samego szumu, w T krokach) i każemy sieci zgadnąć rysunek spod szumu. Rysowanie: zaczynamy od czystego szumu i T razy robimy mały krok w stronę tego, co sieć zgaduje. Z szumu powoli wyłania się rysunek.',
    why: 'Tak działają dzisiejsze generatory (Stable Diffusion, DALL·E, Midjourney). Łączy zalety poprzedników: uczy się prostym, stabilnym błędem jak VAE, a rysuje ostro, bo rysunek powstaje w wielu krokach i każdy krok poprawia poprzedni. Kategorię czasem (10%) ukrywamy, więc sieć umie też rysować bez podpowiedzi. Przy rysowaniu mieszamy obie odpowiedzi: suwak „siła podpowiedzi” (classifier-free guidance) mocniej ciągnie rysunek w stronę kategorii.',
    lim: 'Rysowanie jest wolne: T przejść sieci na jeden rysunek (tu 100, w dużych modelach 20–1000). Tutaj sieć jest zwykłą siecią gęstą, a duże modele używają sieci splotowych (U-Net) albo transformerów, które widzą sąsiedztwo pikseli. Opis zdaniem zamiast numeru kategorii wymagałby jeszcze kodera tekstu.',
    more: 'Harmonogram szumu: kosinusowy (Nichol i Dhariwal 2021). Sieć przewiduje czysty rysunek, a nie sam szum jak w oryginalnym DDPM. To ta sama metoda w innej parametryzacji, łatwiejsza dla małej sieci (z przewidywaniem szumu rysunki zostawały zaszumione). Krok t wchodzi do sieci jako wektor sinusów i kosinusów, kategoria jako wyuczony wektor.' },
];
const CPUDEV = { id: 'device', label: 'Liczy', hint: 'jak liczyć', opts: [['wasm', 'procesor, WASM SIMD'], ['cpu', 'procesor, zwykły JS']], val: 'wasm' };
const HP = {
  vae: [
    { id: 'z', label: 'Przestrzeń ukryta', hint: 'liczb z', opts: [2, 8, 16], val: 8 },
    { id: 'hidden', label: 'Warstwa ukryta', hint: 'neuronów', opts: [128, 256, 512], val: 256 },
    { id: 'beta', label: 'Waga KL', hint: 'gładkość przestrzeni', opts: [.5, 1, 2], val: 1 },
    { id: 'steps', label: 'Kroki nauki', hint: 'po 64 rysunki', opts: [1000, 4000, 10000], val: 4000 },
    { id: 'lr', label: 'Tempo nauki', hint: 'wielkość kroku', opts: [.0003, .001, .003], val: .001 }, { ...CPUDEV },
  ],
  gan: [
    { id: 'noise', label: 'Szum wejściowy', hint: 'liczb', opts: [8, 16, 32], val: 16 },
    { id: 'hidden', label: 'Warstwy ukryte', hint: 'neuronów', opts: [128, 256, 512], val: 256 },
    { id: 'steps', label: 'Kroki gry', hint: 'po 64 rysunki', opts: [2000, 6000, 15000], val: 6000 },
    { id: 'lr', label: 'Tempo nauki', hint: 'generatora', opts: [.0001, .0002, .0005], val: .0002 }, { ...CPUDEV },
  ],
  diff: [
    { id: 'T', label: 'Kroki szumu', hint: 'T', opts: [50, 100, 200], val: 100 },
    { id: 'hidden', label: 'Warstwy ukryte', hint: 'neuronów (×3)', opts: [256, 512], val: 512 },
    { id: 'steps', label: 'Kroki nauki', hint: 'po 64 rysunki', opts: [1000, 4000, 10000], val: 4000 },
    { id: 'lr', label: 'Tempo nauki', hint: 'wielkość kroku', opts: [.0003, .001, .002], val: .001 }, { ...CPUDEV },
  ],
  ar: [
    { id: 'layers', label: 'Warstwy', hint: 'bloków uwagi', opts: [1, 2, 3, 4], val: 2 },
    { id: 'dim', label: 'Szerokość', hint: 'liczb na piksel', opts: [32, 48, 64], val: 48 },
    { id: 'heads', label: 'Głowy', hint: 'uwagi na warstwę', opts: [2, 4], val: 4 },
    { id: 'steps', label: 'Kroki nauki', hint: 'po 8 rysunków', opts: [1000, 3000, 6000, 12000], val: 3000 },
    { id: 'lr', label: 'Tempo nauki', hint: 'wielkość kroku', opts: [.001, .003], val: .003 },
    { id: 'device', label: 'Liczy', hint: 'jak liczyć', opts: [['wasm', 'procesor, WASM SIMD'], ['cpu', 'procesor, zwykły JS'], ['gpu', 'karta graficzna (GPU)']], val: 'wasm' },
  ],
};
let cur = 'stats', man = null, catData = [], ready = false, training = false;
const hpValues = (kind = cur) => { const o = { batch: kind === 'ar' ? 8 : 64 }; (HP[kind] || []).forEach(h => o[h.id] = h.val); return o; };

// ---------- dane: pliki PNG z img/data, każdy kafelek 16×16 to jeden rysunek ----------
async function loadData() {
  try {
    man = await (await fetch('data/manifest.json', { cache: 'no-cache' })).json();
    catData = await Promise.all(man.categories.map(async c => {
      const im = new Image(); im.src = 'data/' + c.file; await im.decode();
      const cv = document.createElement('canvas'); cv.width = im.width; cv.height = im.height;
      const cx = cv.getContext('2d'); cx.drawImage(im, 0, 0);
      const d = cx.getImageData(0, 0, im.width, im.height).data, pix = new Uint8Array(c.count * N);
      for (let k = 0; k < c.count; k++) {
        const tx = (k % man.cols) * S, ty = Math.floor(k / man.cols) * S;
        for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
          const g = d[((ty + y) * im.width + tx + x) * 4];
          pix[k * N + y * S + x] = Math.max(0, Math.min(LV - 1, Math.round((242 - g) / (242 - 24) * (LV - 1))));
        }
      }
      return { ...c, pix };
    }));
  } catch (e) { $('dataErr').textContent = 'Nie udało się wczytać rysunków z img/data/. Strona musi być otwarta przez serwer (np. GitHub Pages).'; $('dataErr').hidden = false; return; }
  const total = catData.reduce((s, c) => s + c.count, 0);
  $('dataStats').innerHTML = '';
  [['Kategorie', catData.length], ['Rysunki', total.toLocaleString('pl')], ['Rozmiar', '16 × 16'], ['Odcienie', LV], ['Liczb na rysunek', N]].forEach(([k, v]) => {
    const d = document.createElement('div'); d.innerHTML = '<dt></dt><dd></dd>'; d.firstChild.textContent = k; d.lastChild.textContent = v; $('dataStats').appendChild(d);
  });
  $('gallery').innerHTML = '';
  catData.forEach(c => {
    const box = document.createElement('div'), b = document.createElement('b'), sm = document.createElement('small'), cv = document.createElement('canvas');
    box.className = 'gcat'; b.textContent = c.name; sm.textContent = `${c.count} rysunków · ${c.quickdraw}`; b.appendChild(sm);
    cv.width = 10 * (S + 1) - 1; cv.height = S; const cx = cv.getContext('2d');
    for (let k = 0; k < 10; k++) drawImg(cx, c.pix.subarray(k * N, (k + 1) * N), k * (S + 1), 0, 1);
    box.append(b, cv); $('gallery').appendChild(box);
  });
  const ul = document.createElement('ul');
  [['manifest.json', 'lista kategorii i parametry danych'], ['LICENCJA.txt', 'źródło i licencja (CC BY 4.0)'], ...catData.map(c => [c.file, `${c.name}: ${c.count} rysunków w siatce po ${man.cols} w wierszu`])].forEach(([f, d]) => {
    const li = document.createElement('li'), a = document.createElement('a'); a.href = 'data/' + f; a.target = '_blank'; a.rel = 'noopener';
    const code = document.createElement('code'); code.textContent = f; a.appendChild(code); li.append(a, ' — ' + d); ul.appendChild(li);
  });
  const p = document.createElement('p'); p.className = 'note';
  p.textContent = 'Każdy plik PNG to siatka rysunków 16×16, dokładnie to, z czego uczy się sieć. Robi je skrypt img/tools/build_quickdraw.py: pobiera pierwsze 2000 rysunków każdej kategorii z Quick, Draw!, pomniejsza je z 28×28 do 16×16 i sprowadza do 16 odcieni. Ostatnie 10% rysunków każdej kategorii służy tylko do sprawdzania.';
  $('files').replaceChildren(p, ul);
  $('genCat').innerHTML = ''; catData.forEach((c, i) => $('genCat').add(new Option(c.name, i)));
  worker.postMessage({ cmd: 'data', cats: catData.map(c => ({ name: c.name, pix: c.pix })) });
}
function drawImg(cx, pix, ox, oy, sc) {
  for (let i = 0; i < N; i++) { const g = shade(pix[i]); cx.fillStyle = `rgb(${g},${g},${g})`; cx.fillRect(ox + (i % S) * sc, oy + Math.floor(i / S) * sc, sc, sc); }
}

// ---------- wybór kroku i ustawienia ----------
function renderSteps() {
  $('steps').innerHTML = '';
  STEPS.forEach(s => {
    const b = document.createElement('button'); b.type = 'button'; b.setAttribute('role', 'tab'); b.id = 'step-' + s.kind;
    b.setAttribute('aria-selected', s.kind === cur ? 'true' : 'false'); b.disabled = !s.ready;
    b.innerHTML = '<span></span><small></small>'; b.firstChild.textContent = s.name; b.lastChild.textContent = s.sub;
    b.addEventListener('click', () => { cur = s.kind; renderAll(); resetLearn(); });
    $('steps').appendChild(b);
  });
}
function renderExplain() {
  const s = STEPS.find(x => x.kind === cur);
  $('explain').innerHTML = `<div><h4>Jak działa</h4><p>${s.what}</p></div><div><h4>Dlaczego taka konstrukcja</h4><p>${s.why}</p></div><div><h4 class="lim">Czego nie umie</h4><p>${s.lim}</p></div>` + (s.more ? `<div><h4>Szczegóły</h4><p>${s.more}</p></div>` : '');
}
function renderHP() {
  $('hp').innerHTML = '';
  (HP[cur] || []).forEach(h => {
    const lab = document.createElement('label'); lab.htmlFor = 'hp-' + h.id; lab.innerHTML = '<span><b></b><small></small></span>';
    lab.querySelector('b').textContent = h.label; lab.querySelector('small').textContent = h.hint;
    const sel = document.createElement('select'); sel.id = 'hp-' + h.id;
    h.opts.forEach(o => Array.isArray(o) ? sel.add(new Option(o[1], o[0])) : sel.add(new Option(String(o).replace('.', ','), o)));
    if (h.id === 'device' && !('gpu' in navigator)) { sel.options[2].disabled = true; sel.options[2].textContent += ' — niedostępne tutaj'; }
    sel.value = h.val; sel.addEventListener('change', () => { h.val = isNaN(+sel.value) ? sel.value : +sel.value; renderGraph(); if (h.id === 'device' && h.val === 'gpu') checkGPU(); });
    lab.appendChild(sel); $('hp').appendChild(lab);
  });
}
function renderAll() { renderSteps(); renderExplain(); renderHP(); renderGraph(); }

// ---------- graf modelu ----------
// Węzły: nazwa, kształt danych na wyjściu, wagi (kształty i liczba parametrów), opis po kliknięciu.
function graphSpec(kind, hp) {
  const C = catData.length || 12;
  if (kind === 'stats') return { nodes: [
    { id: 'in', name: 'Wejście: kategoria', out: '1 numer (0–' + (C - 1) + ')', io: true, desc: 'Numer kategorii, np. 0 = kot. Model nie dostaje żadnych pikseli: rysuje wyłącznie z tabeli tej kategorii.' },
    { id: 'tab', name: 'Tabela odcieni', out: '256 × 16', w: [['tabela', `${C} × 256 × 16`, C * N * LV]], desc: `Dla każdej z ${C} kategorii i każdego z 256 pikseli 16 prawdopodobieństw odcieni. Policzone wprost z rysunków treningowych, bez uczenia gradientem: to zwykłe zliczanie, jak w bigramie.` },
    { id: 'out', name: 'Losowanie każdego piksela', out: '16 × 16 odcieni', io: true, desc: 'Każdy z 256 pikseli losowany osobno ze swojego rozkładu. Brak zależności między pikselami: dlatego rysunek wychodzi jak szum w kształcie kategorii.' },
  ] };
  const dl = (id, name, nIn, nOut, out, desc, act) => ({ id, name: name + (act ? ` · ${act}` : ''), out, w: [['W', `${nIn} × ${nOut}`, nIn * nOut], ['bias', `${nOut}`, nOut]], desc });
  if (kind === 'vae') { const Hh = hp.hidden, Z = hp.z; return { nodes: [
    { id: 'in', name: 'Wejście: rysunek + kategoria', out: `${N} + ${C} liczb`, io: true, desc: `256 odcieni (0–1) i kategoria jako ${C} liczb, z których jedna to 1 (one-hot).` },
    dl('e1', 'Koder: warstwa ukryta', N + C, Hh, `${Hh}`, 'Zbiera cechy rysunku.', 'tanh'),
    { id: 'z', name: `Średnia mu i rozrzut z (2 × ${Z})`, out: `${Z} liczb z`, w: [['W mu', `${Hh} × ${Z}`, Hh * Z], ['bias mu', `${Z}`, Z], ['W log-wariancji', `${Hh} × ${Z}`, Hh * Z], ['bias', `${Z}`, Z]], desc: `Koder nie podaje jednego punktu, tylko rozkład: środek mu i rozrzut. z = mu + rozrzut × losowy szum (reparametryzacja). Te ${Z} liczb to cały „opis” rysunku.` },
    { id: 'kl', name: 'Błąd KL: z ma być jak N(0, 1)', out: '1 liczba', io: true, desc: 'Kara za to, że rozkład z odbiega od zwykłego rozkładu normalnego. Dzięki niej po nauce można losować z z N(0, 1) i rysować.' },
    dl('d1', 'Dekoder: z + kategoria → ukryta', Z + C, Hh, `${Hh}`, 'Z kilku liczb i kategorii zaczyna odtwarzać rysunek.', 'tanh'),
    dl('d2', 'Dekoder: wyjście', Hh, N, `${N}`, 'Dla każdego piksela jedna liczba, po sigmoidzie odcień 0–1. Błąd odtworzenia: binarna entropia krzyżowa z prawdziwym rysunkiem.', 'sigmoid'),
    { id: 'out', name: 'Rysunek', out: '16 × 16', io: true, desc: 'Wszystkie piksele naraz, w jednym przejściu. Rysując nowe, losujemy z z N(0, 1) i używamy tylko dekodera.' },
  ] }; }
  if (kind === 'gan') { const Hh = hp.hidden, Z = hp.noise; return { nodes: [
    { id: 'in', name: 'Generator: szum + kategoria', out: `${Z} + ${C} liczb`, io: true, desc: `Losowe ${Z} liczb i kategoria. Inny szum = inny rysunek tej samej kategorii.` },
    dl('g1', 'Generator: warstwa 1', Z + C, Hh, `${Hh}`, 'Rozwija szum w cechy rysunku.', 'GELU'),
    dl('g2', 'Generator: warstwa 2', Hh, Hh, `${Hh}`, '', 'GELU'),
    dl('g3', 'Generator: wyjście', Hh, N, `${N}`, 'Odcień każdego piksela (sigmoid). To jest podróbka.', 'sigmoid'),
    dl('k1', 'Krytyk: rysunek + kategoria', N + C, Hh, `${Hh}`, 'Krytyk dostaje raz prawdziwy rysunek, raz podróbkę generatora.', 'GELU'),
    dl('k2', 'Krytyk: warstwa 2', Hh, Hh >> 1, `${Hh >> 1}`, '', 'GELU'),
    { id: 'k3', name: 'Krytyk: ocena (+ projekcja kategorii)', out: '1 liczba', w: [['W', `${Hh >> 1} × 1`, Hh >> 1], ['bias', '1', 1], ['wektory kategorii', `${C} × ${Hh >> 1}`, C * (Hh >> 1)]], desc: 'Jedna liczba: jak bardzo krytyk wierzy, że rysunek jest prawdziwy i pasuje do kategorii. Krytyk chce: prawdziwe → 1, podróbki → 0. Generator chce, żeby jego podróbki dostawały 1.' },
    { id: 'out', name: 'Gra: krytyk ↔ generator', out: 'dwa błędy', io: true, desc: 'Na zmianę krok krytyka i krok generatora. Błąd generatora liczony jest przez krytyka: gradient płynie przez krytyka do generatora, ale uczy się tylko generator.' },
  ] }; }
  if (kind === 'diff') { const Hh = hp.hidden; return { nodes: [
    { id: 'in', name: 'Wejście: zaszumiony rysunek', out: `${N} liczb`, io: true, desc: `Rysunek z dodanym szumem w kroku t (od 1 do ${hp.T}). Przy t = ${hp.T} to prawie sam szum.` },
    { id: 'te', name: '+ krok t i kategoria', out: `${N} + 32 + 32`, w: [['wektory kategorii (+ „brak”)', `${C + 1} × 32`, (C + 1) * 32]], desc: 'Krok t jako 32 sinusy i kosinusy (bez wag do nauki), kategoria jako wyuczony wektor 32 liczb. Dodatkowy wektor „brak kategorii” do rysowania bez podpowiedzi.' },
    dl('l1', 'Warstwa 1', N + 64, Hh, `${Hh}`, '', 'GELU'), dl('l2', 'Warstwa 2', Hh, Hh, `${Hh}`, '', 'GELU'), dl('l3', 'Warstwa 3', Hh, Hh, `${Hh}`, '', 'GELU'),
    dl('l4', 'Wyjście: zgadnięty rysunek', Hh, N, `${N}`, 'Sieć zgaduje czysty rysunek spod szumu. Błąd: średni kwadrat różnicy z prawdziwym rysunkiem.'),
    { id: 'out', name: `Rysowanie: ${hp.T} małych kroków`, out: '16 × 16', io: true, desc: `Start z czystego szumu. W każdym z ${hp.T} kroków sieć zgaduje rysunek (dwa razy: z kategorią i bez, mieszane siłą podpowiedzi), a my przesuwamy szum trochę w jego stronę i dodajemy odrobinę nowego szumu.` },
  ] }; }
  const d = hp.dim, L = hp.layers, nh = hp.heads, Vin = LV + C;
  const blk = [
    { id: 'ln1', name: 'Normalizacja warstwy', out: `256 × ${d}`, w: [['skala g', `${d}`, d], ['przesunięcie b', `${d}`, d]], desc: 'Każdy wektor pozycji sprowadzony do średniej 0 i wariancji 1, potem przeskalowany. Trzyma liczby w rozsądnym zakresie przy wielu warstwach.' },
    { id: 'att', name: `Uwaga przyczynowa (${nh} głowy)`, out: `256 × ${d}`, w: [['Wqkv', `${d} × ${3 * d}`, 3 * d * d], ['bias qkv', `${3 * d}`, 3 * d], ['Wo (wyjście uwagi)', `${d} × ${d}`, d * d], ['bias wyjścia', `${d}`, d]],
      desc: `Każda pozycja liczy zapytanie, klucz i wartość (3 × ${d} liczb). ${nh} głowy po ${d / nh} liczb niezależnie decydują, na które wcześniejsze piksele patrzeć. „Przyczynowa”: piksel widzi tylko piksele przed sobą, nigdy po. Tu sieć odkrywa, że piksel nad nim jest 16 pozycji wcześniej.`, res: true },
    { id: 'ln2', name: 'Normalizacja warstwy', out: `256 × ${d}`, w: [['skala g', `${d}`, d], ['przesunięcie b', `${d}`, d]], desc: 'Jak wyżej, przed siecią w każdej pozycji.' },
    { id: 'mlp', name: `Sieć w pozycji: ${d} → ${4 * d} → ${d}`, out: `256 × ${d}`, w: [['W1', `${d} × ${4 * d}`, 4 * d * d], ['bias 1', `${4 * d}`, 4 * d], ['W2', `${4 * d} × ${d}`, 4 * d * d], ['bias 2', `${d}`, d]],
      desc: 'Dwie warstwy z funkcją GELU, osobno w każdej pozycji. Tu siedzi większość parametrów: przetwarza to, co uwaga zebrała z innych pikseli.', res: true },
  ];
  return { nodes: [
    { id: 'in', name: 'Wejście: kategoria + 255 pikseli', out: '256 numerów', io: true, desc: `Ciąg 256 liczb: najpierw numer kategorii (${LV}–${Vin - 1}), potem odcienie pikseli 0–254 (0–15), wiersz po wierszu. Ostatni piksel nie wchodzi, bo nie ma już czego po nim zgadywać.` },
    { id: 'emb', name: 'Embedding', out: `256 × ${d}`, w: [['tabela wektorów', `${Vin} × ${d}`, Vin * d]], desc: `Każdy z ${Vin} „wyrazów” (16 odcieni i ${C} kategorii) ma własny wektor ${d} liczb. Tak numer kategorii zamienia się w coś, z czego sieć może korzystać, bez rozumienia słowa „kot”.` },
    { id: 'pos', name: '+ pozycja', out: `256 × ${d}`, w: [['wektory pozycji', `256 × ${d}`, N * d]], desc: 'Do wektora każdego piksela dodajemy wyuczony wektor jego miejsca (0–255). Bez tego uwaga nie wiedziałaby, który piksel jest który.' },
    { group: true, repeat: L, nodes: blk },
    { id: 'lnf', name: 'Normalizacja końcowa', out: `256 × ${d}`, w: [['skala g', `${d}`, d], ['przesunięcie b', `${d}`, d]], desc: 'Ostatnie wyrównanie przed wyjściem.' },
    { id: 'out', name: 'Wyjście: 16 odcieni', out: '256 × 16', w: [['W', `${d} × 16`, d * LV], ['bias', '16', LV]], desc: 'Dla każdej z 256 pozycji 16 liczb, po softmaksie prawdopodobieństwa odcieni następnego piksela. Błąd = −ln(prawdopodobieństwa prawdziwego odcienia), średnio po pikselach.' },
    { id: 'smp', name: 'Softmax → losowanie', out: 'następny piksel', io: true, desc: 'Przy rysowaniu losujemy odcień z tego rozkładu (z temperaturą), dopisujemy go do ciągu i liczymy dalej. 256 razy.' },
  ] };
}
let selNode = null, graphView = 'blocks', wviz = null;
function renderGraphTabs() {
  document.querySelector('.graphrow').classList.toggle('wide', graphView === 'neurons');
  const box = $('graphTabs'); box.innerHTML = '';
  [['blocks', 'Bloki i kształty'], ['neurons', 'Neurony i połączenia']].forEach(([id, nm]) => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = nm; b.setAttribute('aria-pressed', graphView === id ? 'true' : 'false');
    b.addEventListener('click', () => { graphView = id; renderGraph(); }); box.appendChild(b);
  });
}
function renderGraph() {
  renderGraphTabs();
  const spec = graphSpec(cur, hpValues()), svg = $('graph'), W = 600, bw = 300, bh = 44, gap = 30, x0 = 64;
  let y = 14, h = '', rows = [], total = 0;
  const box = (n, yy, rep) => {
    const p = (n.w || []).reduce((s, w) => s + w[2], 0) * (rep || 1);
    h += `<g class="node${n.io ? ' io' : ''}${selNode === n.id ? ' sel' : ''}" data-id="${n.id}" tabindex="0" role="button" aria-label="${n.name}">`
      + `<rect x="${x0}" y="${yy}" width="${bw}" height="${bh}" rx="3"/>`
      + `<text x="${x0 + 12}" y="${yy + 19}">${n.name}</text>`
      + (p ? `<text class="p" x="${x0 + 12}" y="${yy + 35}">${p.toLocaleString('pl')} parametrów${rep > 1 ? ` (${rep} × ${(p / rep).toLocaleString('pl')})` : ''}</text>` : '')
      + `</g>`;
    if (!n.last) h += `<text class="shape" x="${x0 + bw + 14}" y="${yy + bh + gap / 2 + 4}">${n.out}</text>`;
    return p;
  };
  const arrow = (yy) => { h += `<path class="edge" d="M${x0 + bw / 2} ${yy} v${gap - 6}" marker-end="url(#ar)"/>`; };
  spec.nodes[spec.nodes.length - 1].last = true;
  let y0 = y; y += 16;
  spec.nodes.forEach((n, i) => {
    if (n.group) {
      const top = y - 8; let first = true, resStart = null;
      n.nodes.forEach((b, j) => {
        if (b.id === 'ln1' || b.id === 'ln2') resStart = y - gap / 2;
        total += box(b, y, n.repeat); rows.push({ n: b, rep: n.repeat });
        if (b.res) { const yy = y + bh + gap / 2; h += `<path class="res" d="M${x0} ${resStart} h-16 V${yy} h${16 + bw / 2 - 4}" marker-end="url(#ar2)"/><text class="shape" x="${x0 - 28}" y="${(resStart + yy) / 2}" text-anchor="end">+</text>`; }
        y += bh; arrow(y); y += gap; first = false;
      });
      h += `<rect class="frame" x="${x0 - 22}" y="${top}" width="${bw + 44}" height="${y - top - gap + 8}"/>`
        + `<text class="frametxt" x="${x0 - 18}" y="${top - 6}">blok × ${n.repeat} (powtórzony)</text>`;
    } else {
      total += box(n, y, 1); rows.push({ n, rep: 1 }); y += bh;
      if (i < spec.nodes.length - 1) { arrow(y); y += gap; }
    }
  });
  y += 14;
  svg.setAttribute('viewBox', `0 0 ${W} ${y}`); svg.style.maxWidth = svg.style.minWidth = svg.style.margin = '';
  svg.innerHTML = `<defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="#9a9aa6"/></marker>`
    + `<marker id="ar2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="#c9a86a"/></marker></defs>` + h;
  svg.querySelectorAll('.node').forEach(g => {
    const pickN = () => { selNode = g.dataset.id; renderGraph(); };
    g.addEventListener('click', pickN); g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickN(); } });
  });
  // opis wybranego węzła i tabela parametrów
  const sel = rows.find(r => r.n.id === selNode) || rows[0];
  const info = $('graphInfo'); info.innerHTML = '<h4></h4><p></p>';
  info.querySelector('h4').textContent = sel.n.name + (sel.rep > 1 ? ` (w każdym z ${sel.rep} bloków)` : '');
  info.querySelector('p').textContent = sel.n.desc + ` Wyjście: ${sel.n.out}.`;
  const tb = $('ptable').querySelector('tbody'); tb.innerHTML = '';
  rows.forEach(({ n, rep }) => (n.w || []).forEach(([nm, sh, p]) => {
    const tr = document.createElement('tr'), pt = p * rep;
    [`${n.name} · ${nm}${rep > 1 ? ` (×${rep})` : ''}`, sh, pt.toLocaleString('pl')].forEach((v, i) => { const td = document.createElement('td'); td.textContent = v; if (i === 2) { const s = document.createElement('span'); s.className = 'bar'; s.style.width = Math.max(2, 60 * pt / total) + 'px'; td.appendChild(s); } tr.appendChild(td); });
    tb.appendChild(tr);
  }));
  const tr = document.createElement('tr'); tr.className = 'sum'; ['Razem', '', total.toLocaleString('pl')].forEach(v => { const td = document.createElement('td'); td.textContent = v; tr.appendChild(td); }); tb.appendChild(tr);
  $('paramNote').textContent = `${total.toLocaleString('pl')} parametrów · ${(total * 4 / 1024).toFixed(0)} KB`;
  if (graphView === 'neurons') renderNeurons();
}

// ---------- karta graficzna: test zgodności przy pierwszym wyborze ----------
let gpuState = null;
function checkGPU() {
  if (gpuState && gpuState.ok) return gpuNote();
  if (gpuState === 'testing') return;
  gpuState = 'testing'; $('train').disabled = true; $('paramNote').textContent = 'Sprawdzam kartę graficzną: ten sam krok nauki na CPU i GPU…';
  worker.postMessage({ cmd: 'gputest' });
}
function gpuNote() {
  if (!gpuState || gpuState === 'testing') return;
  $('gpuNote').hidden = false;
  $('gpuNote').textContent = gpuState.ok
    ? `Karta: ${gpuState.name}. Test zgodności: ten sam krok nauki na CPU i GPU dał błąd ${fmt(gpuState.lossC, 5)} i ${fmt(gpuState.lossG, 5)}, wagi po kroku różnią się najwyżej o ${gpuState.diff.toExponential(1).replace('.', ',')} (zaokrąglenia). Przy obrazkach (256 pozycji naraz) karta zwykle wyraźnie wygrywa z procesorem: porównaj czas kroku w tabeli.`
    : `Karta graficzna niedostępna: ${gpuState.msg}. Liczę na procesorze.`;
}

// Neurony: kolumna kółek na warstwę (dla jednej pozycji, czyli jednego piksela). Szerokie warstwy pokazujemy
// próbką kółek z liczbą przy kolumnie. Linie to wagi: zielone dodatnie, niebieskie ujemne, grubsze silniejsze.
function renderNeurons() {
  const svg = $('graph'), hp = hpValues(), C = catData.length || 12;
  if (cur === 'stats') {
    const cols = [{ n: C, show: Math.min(C, 12), label: 'kategoria', sub: `${C} wejść` }, { n: LV, show: LV, label: 'odcienie piksela', sub: '16 wyjść × 256 pikseli' }];
    return drawNeuronCols(svg, cols, [{ type: 'full', note: 'tabela' }], 'Statystyka nie ma warstw ukrytych: każda kategoria ma po prostu własną tabelę odcieni dla każdego z 256 miejsc. To jedno „połączenie” na parę kategoria → odcień w każdym miejscu.');
  }
  if (cur !== 'ar') {
    const wv = wviz && wviz.cols ? wviz : null;
    if (!wv) { $('graph').innerHTML = ''; const info = $('graphInfo'); info.innerHTML = '<h4>Neurony i połączenia</h4><p>Kliknij „Ucz”: widok pojawi się z pierwszymi wagami sieci.</p>'; return; }
    return drawNeuronCols(svg, wv.cols.map(c => ({ ...c, sub: String(c.n) })), wv.mats.map(M => ({ type: 'w', M })), 'Wagi z aktualnego stanu sieci (próbka kółek i połączeń). Zielone dodatnie, niebieskie ujemne, grubsze silniejsze.' + (cur === 'gan' ? ' Pokazany jest generator; krytyk ma podobną budowę w odwrotnym kierunku.' : ''));
  }
  const d = hp.dim, L = hp.layers, cols = [{ n: LV + C, show: 12, label: 'wejście', sub: `1 z ${LV + C}`, labels: wviz && wviz.tok }, { n: d, show: 12, label: 'embedding + pozycja', sub: `${d}` }], links = [{ type: 'w', key: 'emb' }];
  for (let l = 0; l < L; l++) {
    cols.push({ n: d, show: 12, label: `uwaga ${l + 1}`, sub: `${d} · z innych pikseli` }); links.push({ type: 'att' });
    cols.push({ n: 4 * d, show: 16, label: `sieć ${l + 1}: warstwa ukryta`, sub: `${4 * d} · GELU` }); links.push({ type: 'w', key: 'w1', l });
    cols.push({ n: d, show: 12, label: `wynik bloku ${l + 1}`, sub: `${d}` }); links.push({ type: 'w', key: 'w2', l });
  }
  cols.push({ n: LV, show: LV, label: 'wyjście', sub: '16 odcieni', shades: true }); links.push({ type: 'w', key: 'wo' });
  drawNeuronCols(svg, cols, links, wviz ? 'Wagi z aktualnego stanu sieci. Pokazana jest próbka kółek (np. 12 z ' + d + '), więc i połączeń. Przerywane linie przy uwadze to „zbierz informacje z wcześniejszych pikseli”: ta warstwa łączy różne pozycje, reszta działa w każdej pozycji osobno. Połączenia omijające (dodawanie wejścia bloku do wyniku) pominięte dla czytelności.' : 'Po rozpoczęciu nauki linie pokażą prawdziwe wagi: zielone dodatnie, niebieskie ujemne, grubsze silniejsze.');
}
function drawNeuronCols(svg, cols, links, note) {
  const colW = 110, H = 500, top = 70, bot = 56, x0 = 90, W = x0 * 2 + colW * (cols.length - 1), r = 6.5;
  const pos = cols.map((c, ci) => { const n = c.show, gap = Math.min(26, (H - top - bot) / n); return Array.from({ length: n }, (_, i) => [x0 + ci * colW, top + (H - top - bot - gap * (n - 1)) / 2 + i * gap]); });
  let h = '';
  links.forEach((lk, li) => {
    const A = pos[li], B = pos[li + 1];
    let M = null;
    if (lk.M) M = lk.M;
    else if (lk.type === 'w' && wviz && wviz.emb) M = lk.key === 'emb' ? wviz.emb : lk.key === 'wo' ? wviz.wo : wviz.blocks[lk.l] && wviz.blocks[lk.l][lk.key];
    const mx = M ? Math.max(1e-9, ...M.flat().map(Math.abs)) : 1;
    A.forEach(([ax, ay], i) => B.forEach(([bx, by], j) => {
      if (lk.type === 'att') { if (i === j || (i + 3) % A.length === j) h += `<line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}" stroke="#9a9aa6" stroke-width=".8" stroke-dasharray="3 3" opacity=".55"/>`; return; }
      if (M && M[i] && M[i][j] != null) { const v = M[i][j] / mx, a = Math.abs(v); if (a < .08) return; h += `<line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}" stroke="${v > 0 ? '#9cac70' : '#7fa6d8'}" stroke-width="${(.4 + a * 2.2).toFixed(2)}" opacity="${(.15 + a * .75).toFixed(2)}"/>`; }
      else h += `<line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}" stroke="#55556a" stroke-width=".5" opacity=".35"/>`;
    }));

  });
  cols.forEach((c, ci) => {
    pos[ci].forEach(([x, y], i) => {
      const fill = c.shades ? `rgb(${shade(i)},${shade(i)},${shade(i)})` : 'var(--bg)';
      h += `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="var(--rot)" stroke-width="1.2"/>`;
      if (c.labels && c.labels[i]) h += `<text class="shape" x="${x - 10}" y="${y + 4}" text-anchor="end">${c.labels[i]}</text>`;
    });
    if (c.n > c.show) h += `<text class="shape" x="${pos[ci][0][0]}" y="${pos[ci][pos[ci].length - 1][1] + 20}" text-anchor="middle">⋮ ${c.n}</text>`;
    const up = ci % 2 ? 32 : 14;                       // podpisy na przemian wyżej i niżej, żeby na siebie nie nachodziły
    h += `<text class="frametxt" x="${pos[ci][0][0]}" y="${top - up}" text-anchor="middle" font-size="11">${c.label}</text>`;
    h += `<text class="shape" x="${pos[ci][0][0]}" y="${H - (ci % 2 ? 6 : 22)}" text-anchor="middle">${c.sub}</text>`;
  });
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = h; svg.style.maxWidth = Math.round(W * 1.25) + 'px'; svg.style.minWidth = Math.min(760, Math.round(W * .9)) + 'px'; svg.style.margin = '0 auto';
  const info = $('graphInfo'); info.innerHTML = '<h4>Neurony i połączenia</h4><p></p>'; info.querySelector('p').textContent = note;
}

// ---------- nauka ----------
let series = { train: [], val: [] }, live = null, att = null, attSel = { head: 0, pix: 136 };
const runs = [], have = new Set();
const BITS = k => k === 'stats' || k === 'ar';
const LABEL = (k, hp) => ({ stats: '1 · Statystyka pikseli', ar: `2 · Piksel po pikselu (${hp.layers}×${hp.dim}, ${hp.heads} gł.)`, vae: `3 · VAE (z = ${hp.z}, ${hp.hidden})`, gan: `4 · GAN (${hp.hidden})`, diff: `5 · Dyfuzja (T = ${hp.T}, ${hp.hidden})` }[k]);
function resetLearn() {
  series = { train: [], val: [] }; live = null; drawChart(); renderMetrics(null);
  $('samples').innerHTML = ''; $('sampNote').textContent = 'Kliknij „Ucz”.';
  $('gen').disabled = !have.has(cur);
  $('attTitle').hidden = $('attCtl').hidden = cur !== 'ar'; $('att').parentNode.hidden = cur !== 'ar';
  $('tempBox').hidden = !BITS(cur); $('guideBox').hidden = cur !== 'diff';
  $('legT').textContent = cur === 'gan' ? 'błąd generatora' : cur === 'vae' ? 'błąd na piksel, rysunki treningowe' : cur === 'diff' ? 'błąd zgadywania rysunku (trening)' : 'błąd na rysunkach treningowych';
  $('legV').textContent = cur === 'gan' ? 'błąd krytyka' : cur === 'diff' ? 'błąd zgadywania rysunku (sprawdzające)' : cur === 'vae' ? 'błąd na piksel, rysunki sprawdzające' : 'błąd na rysunkach sprawdzających';
  renderInner();
}
$('train').addEventListener('click', () => {
  if (!ready) return;
  series = { train: [], val: [] }; training = true;
  $('train').disabled = true; $('stop').disabled = cur === 'stats';
  $('sampNote').textContent = cur === 'stats' ? 'Liczę…' : `Sieć zaczyna od losowych wag. Próbki rysunków pojawiają się co ${cur === 'ar' ? 'ok. 20' : 'kilka'} s.`;
  worker.postMessage({ cmd: 'train', kind: cur, hp: hpValues() });
});
$('stop').addEventListener('click', () => worker.postMessage({ cmd: 'stop' }));
$('temp').addEventListener('input', () => $('tempOut').textContent = (+$('temp').value).toFixed(1).replace('.', ','));
$('guide').addEventListener('input', () => $('guideOut').textContent = (+$('guide').value).toFixed(1).replace('.', ','));
let genId = 0;
$('gen').addEventListener('click', () => {
  $('gen').disabled = true; const cx = $('draw').getContext('2d'); cx.fillStyle = '#2a2a33'; cx.fillRect(0, 0, 192, 192);
  worker.postMessage({ cmd: 'generate', kind: cur, classes: [+$('genCat').value], temp: +$('temp').value, guide: +$('guide').value, id: ++genId });
});
function showErr(msg) { $('err').textContent = 'Błąd: ' + msg; $('err').hidden = false; }

worker.onmessage = e => {
  const m = e.data;
  if (m.type === 'ready') { ready = true; $('train').disabled = false; return; }
  if (m.type === 'gputest') {
    gpuState = m; $('train').disabled = training || !ready; renderGraph();
    if (!m.ok) { const h = HP.ar.find(x => x.id === 'device'); h.val = 'wasm'; if ($('hp-device')) $('hp-device').value = 'wasm'; }
    gpuNote(); return;
  }
  if (m.type === 'error') { showErr(m.msg); training = false; $('train').disabled = false; $('stop').disabled = true; $('gen').disabled = !have.has(cur); return; }
  if (m.type === 'gen') {
    if (m.id !== genId) return;
    const cx = $('draw').getContext('2d'), img = m.imgs[0];
    for (let i = 0; i < m.n; i++) { const g = shade(img[i]); cx.fillStyle = `rgb(${g},${g},${g})`; cx.fillRect((i % S) * 12, Math.floor(i / S) * 12, 12, 12); }
    $('drawNote').textContent = cur === 'diff' ? (m.done ? 'Gotowe: rysunek wyłonił się z szumu.' : 'Odszumianie: z czystego szumu krok po kroku wyłania się rysunek.') : BITS(cur) ? 'Sieć rysuje piksel po pikselu, wiersz po wierszu od lewego górnego rogu. Na każdym kroku widzi tylko kategorię i to, co już narysowała.' : 'Wszystkie piksele naraz, w jednym przejściu sieci, z nowego losowego szumu. Kliknij jeszcze raz: inny rysunek tej samej kategorii.';
    if (m.done) $('gen').disabled = false;
    return;
  }
  if (m.type === 'latent') { drawGrid(m.imgs, m.n); return; }
  if (m.type !== 'progress' && m.type !== 'done') return;
  if (m.kind !== cur) return;
  if (m.train != null) series.train.push([m.step, m.train]);
  const v2 = m.kind === 'gan' ? (m.stats && m.stats.d) : m.val;
  if (v2 != null && (!series.val.length || series.val[series.val.length - 1][0] !== m.step)) series.val.push([m.step, v2]);
  live = m; drawChart(); renderMetrics(m);
  if (m.samples) renderSamples(m);
  if (m.attn) { att = m.attn; drawAtt(); }
  if (m.frames) { lastFrames = m.frames; renderInner(); }
  if (m.wviz) { wviz = m.wviz; if (graphView === 'neurons') renderGraph(); }
  if (m.type === 'done') {
    training = false; have.add(m.kind); $('train').disabled = false; $('stop').disabled = true; $('gen').disabled = false;
    runs.push({ kind: m.kind, label: LABEL(m.kind, hpValues(m.kind)), params: m.params, val: m.val, stats: m.stats, ms: m.ms, device: m.device });
    renderCmp(); renderInner();
  }
};
function renderMetrics(m) {
  const v = m ? m.val : null, st = (m && m.stats) || {}, k = cur;
  const krok = ['Krok', m ? m.step.toLocaleString('pl') : '–', m ? `${m.params.toLocaleString('pl')} parametrów` + (m.ms ? ` · ${fmt(m.ms, 0)} ms/krok (${m.device})` : '') + (m.bestStep != null && m.step > 1 ? ` · najlepsza wersja: krok ${m.bestStep.toLocaleString('pl')}` : '') : ''];
  const rows = BITS(k) ? [
    ['Bity na piksel', fmt(v != null ? v / LN2 : null), m && m.train != null ? `treningowe ${fmt(m.train / LN2)} · bez wiedzy: 4,00` : 'bez wiedzy: 4,00'],
    ['Waha się między', v != null ? `${fmt(Math.exp(v), 1)} odcieniami` : '–', 'z 16 możliwych'], krok]
  : k === 'vae' ? [
    ['Błąd na piksel', fmt(v, 4), 'sprawdzający: odtworzenie + KL, mniej = lepiej'],
    ['Odtworzenie i KL', st.rec != null ? `${fmt(st.rec, 3)} + ${fmt(st.kl / N, 3)}` : '–', st.kl != null ? `KL całego rysunku: ${fmt(st.kl, 1)} (ile informacji mieści z)` : 'rozmycie kontra gładkość przestrzeni'], krok]
  : k === 'gan' ? [
    ['Krytyk ocenia', st.pr != null ? `${Math.round(st.pr * 100)}% / ${Math.round(st.pf * 100)}%` : '–', 'prawdziwe / podróbki: ile „prawdziwości” daje krytyk. Równowaga to oba blisko 50%'],
    ['Błędy w grze', st.main != null ? `G ${fmt(st.main, 2)} · K ${fmt(st.d, 2)}` : '–', 'generator i krytyk; to wynik gry, nie jakość rysunków'], krok]
  : [
    ['Błąd zgadywania', fmt(v, 4), 'średni kwadrat różnicy zgadniętego i prawdziwego rysunku (skala −1…1)'],
    ['Na treningu', fmt(m && m.train, 4), 'po losowych krokach szumu t'], krok];
  $('metrics').innerHTML = rows.map(() => '<div><dt></dt><dd></dd></div>').join('');
  [...$('metrics').children].forEach((d, i) => { d.querySelector('dt').textContent = rows[i][0]; const dd = d.querySelector('dd'); dd.textContent = rows[i][1]; const s = document.createElement('small'); s.textContent = rows[i][2]; dd.appendChild(s); });
}
function renderSamples(m) {
  const box = $('samples'); box.innerHTML = '';
  const add = (pix, label) => { const d = document.createElement('div'), cv = document.createElement('canvas'), sp = document.createElement('span'); d.className = 'samp'; cv.width = cv.height = S; drawImg(cv.getContext('2d'), pix, 0, 0, 1); sp.textContent = label; d.append(cv, sp); box.appendChild(d); };
  const head = t => { const h = document.createElement('h4'); h.textContent = t; box.appendChild(h); };
  if (m.mean) m.mean.forEach((pix, c) => add(pix.map(v => Math.round(v)), `${catData[c].name}: średni`));
  const cls = m.sampleClasses || m.samples.map((_, i) => i);
  if (m.recon) head('Nowe rysunki (losowe z)');
  m.samples.forEach((pix, i) => add(pix, catData[cls[i]].name));
  if (m.recon) { head('Odtwarzanie: oryginał → po ściśnięciu do z i odtworzeniu'); m.recon.real.forEach((p, i) => { add(p, 'oryginał'); add(m.recon.rec[i], 'odtworzony'); }); }
  $('sampNote').textContent = {
    stats: 'Najpierw średni obrazek każdej kategorii (oczekiwany odcień w każdym miejscu), potem po jednym losowym rysunku. Średnie przypominają kategorie, ale losowe rysunki to szum: piksele nie wiedzą o sobie nawzajem.',
    ar: `Po jednym rysunku z każdej kategorii z ostatniego sprawdzenia, temperatura 0,8${m.genMs ? ` (rysowanie wszystkich: ${fmt(m.genMs / 1000, 1)} s)` : ''}. Pierwsze kształty pojawiają się zwykle po ok. 3000 krokach.`,
    vae: 'Po jednym rysunku z każdej kategorii z losowego z, a niżej odtworzenia rysunków sprawdzających. Rozmycie to znak rozpoznawczy VAE: gdy sieć nie jest pewna, gdzie kreska, rysuje średnią.',
    gan: 'Ten sam szum przez całą naukę, po jednym na kategorię: widać, jak zmienia się ten sam rysunek. Na początku szum, potem plamy, z czasem kształty. Gdy wszystkie kategorie wyglądają tak samo, to załamanie trybów.',
    diff: 'Po jednym rysunku z każdej kategorii, siła podpowiedzi 2. Każdy powstał z czystego szumu w ' + hpValues('diff').T + ' krokach odszumiania: zobacz „Odszumianie krok po kroku” niżej.',
  }[m.kind];
}
// Podgląd wnętrza: mapa przestrzeni ukrytej (VAE) albo odszumianie krok po kroku (dyfuzja).
let lastFrames = null;
function renderInner() {
  const show = cur === 'vae' || cur === 'diff';
  ['innerTitle', 'innerCtl', 'innerRow'].forEach(id => $(id).hidden = !show);
  if (!show) return;
  const ctl = $('innerCtl'); ctl.innerHTML = '';
  if (cur === 'vae') {
    $('innerTitle').textContent = 'Przestrzeń ukryta';
    const sel = document.createElement('select'); sel.id = 'latCat'; catData.forEach((c, i) => sel.add(new Option(c.name, i)));
    const b = document.createElement('button'); b.type = 'button'; b.className = 'ghost nav'; b.textContent = 'Narysuj mapę'; b.disabled = !have.has('vae');
    b.addEventListener('click', () => worker.postMessage({ cmd: 'latent', c: +sel.value }));
    ctl.append(sel, b);
    $('innerNote').textContent = 'Siatka 8 × 8: każdy rysunek to dekoder dla innego punktu z. W poziomie zmienia się pierwsza liczba z (od −2,5 do 2,5), w pionie druga, pozostałe są 0. Rysunki przechodzą płynnie jeden w drugi: sieć sama ułożyła w tych liczbach cechy rysunku, np. obrót albo grubość. Najlepiej widać to przy 2 liczbach z.';
    if (!have.has('vae')) clearInner('Najpierw naucz VAE.');
  } else {
    $('innerTitle').textContent = 'Odszumianie krok po kroku';
    $('innerNote').textContent = 'Cztery rysunki (kot, dom, drzewo, ryba), od lewej co ok. ' + Math.round(hpValues('diff').T / 10) + ' kroków odszumiania: od czystego szumu do gotowego rysunku. Na początku sieć ustala ogólny kształt, na końcu szczegóły.';
    if (lastFrames) drawFrames(lastFrames); else clearInner('Pojawi się w trakcie nauki.');
  }
}
function clearInner(t) { const cv = $('inner'); cv.width = 288; cv.height = 60; const cx = cv.getContext('2d'); cx.fillStyle = '#1b1b22'; cx.fillRect(0, 0, 288, 60); cx.fillStyle = '#9a9aa6'; cx.font = '13px system-ui'; cx.fillText(t, 12, 34); }
function drawGrid(imgs, n) { const cv = $('inner'), sc = 2, cell = S * sc + 2; cv.width = cv.height = n * cell; const cx = cv.getContext('2d'); cx.fillStyle = '#1b1b22'; cx.fillRect(0, 0, cv.width, cv.height); imgs.forEach((p, k) => drawImg(cx, p, (k % n) * cell + 1, Math.floor(k / n) * cell + 1, sc)); cv.style.width = Math.min(n * cell * 2, 576) + 'px'; }
function drawFrames(fr) {
  const cv = $('inner'), sc = 2, cell = S * sc + 2, nf = fr.length, rows = fr[0].length; cv.width = nf * cell; cv.height = rows * cell;
  const cx = cv.getContext('2d'); cx.fillStyle = '#1b1b22'; cx.fillRect(0, 0, cv.width, cv.height);
  fr.forEach((imgs, f) => imgs.forEach((p, r) => drawImg(cx, p, f * cell + 1, r * cell + 1, sc))); cv.style.width = Math.min(nf * cell * 2, 900) + 'px';
}

// ---------- wykres ----------
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function drawChart() {
  const W = 640, H = 260, cv = $('chart'), dpr = window.devicePixelRatio || 1; cv.width = W * dpr; cv.height = H * dpr;
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const bits = BITS(cur), L = 46, R = 12, T = 14, B = 30, k = bits ? 1 / LN2 : 1, sT = series.train.map(([a, b]) => [a, b * k]), sV = series.val.map(([a, b]) => [a, b * k]), pts = [...sT, ...sV];
  const top = bits ? Math.max(4.2, ...pts.map(p => p[1])) : Math.max(1e-6, ...pts.map(p => p[1])) * 1.08, lo = bits ? Math.max(0, Math.min(1.5, ...pts.map(p => p[1])) - .2) : Math.max(0, Math.min(...pts.map(p => p[1]), top) * .9), maxX = Math.max(1, ...pts.map(p => p[0]));
  const x = s => L + s / maxX * (W - L - R), y = v => T + (1 - (v - lo) / ((top - lo) || 1)) * (H - T - B);
  c.font = '11px "Atkinson Hyperlegible",system-ui,sans-serif'; c.fillStyle = css('--muted'); c.strokeStyle = css('--line'); c.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const v = lo + (top - lo) * i / 4, yy = y(v); c.beginPath(); c.moveTo(L, yy); c.lineTo(W - R, yy); c.stroke(); c.textAlign = 'right'; c.textBaseline = 'middle'; c.fillText(fmt(v, bits ? 1 : 2), L - 6, yy); }
  c.textAlign = 'center'; c.textBaseline = 'top'; [0, .5, 1].forEach(f => c.fillText(Math.round(maxX * f).toLocaleString('pl'), x(maxX * f), H - B + 8));
  if (bits) { const yb = y(4); c.setLineDash([4, 4]); c.strokeStyle = css('--muted'); c.beginPath(); c.moveTo(L, yb); c.lineTo(W - R, yb); c.stroke(); c.setLineDash([]); c.textAlign = 'left'; c.textBaseline = 'bottom'; c.fillText('bez wiedzy (4 bity na piksel)', L + 6, yb - 3); }
  const line = (arr, col, w) => { if (!arr.length) return; if (arr.length === 1) { c.fillStyle = col; c.beginPath(); c.arc(x(arr[0][0]), y(arr[0][1]), 3, 0, 7); c.fill(); return; } c.strokeStyle = col; c.lineWidth = w; c.beginPath(); arr.forEach((p, i) => i ? c.lineTo(x(p[0]), y(p[1])) : c.moveTo(x(p[0]), y(p[1]))); c.stroke(); };
  line(sT, css('--muted'), 1.4); line(sV, css('--rot'), 2.2);
  if (!pts.length) { c.fillStyle = css('--muted'); c.textAlign = 'center'; c.textBaseline = 'middle'; c.font = '13px "Atkinson Hyperlegible",system-ui,sans-serif'; c.fillText('Tu pojawi się krzywa błędu.', W / 2, H / 2); }
}

// ---------- uwaga na obrazku ----------
// Lewa połowa: rysunek z wybranym pikselem. Prawa: ile uwagi wybrany piksel daje każdemu wcześniejszemu
// (pozycja t w ciągu patrzy na pozycje ≤ t; pozycja 0 to kategoria, pozycja j ≥ 1 to piksel j−1).
function drawAtt() {
  if (!att) return;
  const nH = att.L * att.nh, ctl = $('attCtl'); ctl.innerHTML = '';
  for (let k = 0; k < nH; k++) {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip'; b.textContent = `warstwa ${Math.floor(k / att.nh) + 1}, głowa ${k % att.nh + 1}`;
    b.setAttribute('aria-pressed', k === attSel.head ? 'true' : 'false'); b.addEventListener('click', () => { attSel.head = k; drawAtt(); }); ctl.appendChild(b);
  }
  const cv = $('att'), cx = cv.getContext('2d'), sc = 12, l = Math.floor(attSel.head / att.nh), h = attSel.head % att.nh, A = att.att[l], t = attSel.pix;
  drawImg(cx, att.pix, 0, 0, sc);
  cx.strokeStyle = '#d0605e'; cx.lineWidth = 2; cx.strokeRect((t % S) * sc + 1, Math.floor(t / S) * sc + 1, sc - 2, sc - 2);
  const row = A.subarray(h * N * N + t * N, h * N * N + t * N + N); let mx = 0; for (let j = 0; j <= t; j++) mx = Math.max(mx, row[j]);
  cx.fillStyle = '#1b1b22'; cx.fillRect(192, 0, 192, 192);
  for (let j = 1; j <= t; j++) { const p = j - 1, a = row[j] / (mx || 1); cx.fillStyle = `rgba(156,172,112,${a.toFixed(3)})`; cx.fillRect(192 + (p % S) * sc, Math.floor(p / S) * sc, sc, sc); }
  cx.strokeStyle = '#d0605e'; cx.strokeRect(192 + (t % S) * sc + 1, Math.floor(t / S) * sc + 1, sc - 2, sc - 2);
  const cls = Math.round(row[0] * 100), up = t >= 16 ? Math.round(row[t - 15] * 100) : null, left = t % S ? Math.round(row[t] * 100) : null;
  $('attNote').textContent = `Kliknij piksel na lewym rysunku. Prawa strona: ile uwagi daje mu każdy wcześniejszy piksel (jaśniej = więcej), czyli na co sieć patrzy, zgadując zaznaczony piksel. Na kategorię: ${cls}%` + (left != null ? `, na piksel po lewej: ${left}%` : '') + (up != null ? `, na piksel nad nim: ${up}%` : '') + '. Wiele głów uczy się patrzeć właśnie w lewo i w górę: sama odkrywa, że obrazek ma dwa wymiary.';
}
$('att').addEventListener('click', e => {
  const r = e.currentTarget.getBoundingClientRect(), x = (e.clientX - r.left) * 384 / r.width, y = (e.clientY - r.top) * 192 / r.height;
  const px = Math.floor((x % 192) / 12), py = Math.floor(y / 12); if (px < 0 || py < 0 || px > 15 || py > 15) return;
  attSel.pix = py * S + px; drawAtt();
});

// ---------- porównanie ----------
// Bity na piksel mają sens tylko dla modeli, które podają prawdopodobieństwo każdego odcienia (kroki 1–2).
// VAE, GAN i dyfuzja mierzą co innego, więc mają własną miarę; porównuje się je patrząc na rysunki.
function renderCmp() {
  const tb = $('cmp').querySelector('tbody'); tb.innerHTML = '';
  if (!runs.length) { tb.innerHTML = '<tr><td colspan="5" class="note">Tu trafi każda zakończona nauka.</td></tr>'; return; }
  const bitRuns = runs.filter(r => BITS(r.kind)), best = bitRuns.length ? Math.min(...bitRuns.map(r => r.val)) : null;
  runs.forEach(r => {
    const tr = document.createElement('tr'); if (BITS(r.kind) && r.val === best) tr.className = 'best';
    const other = BITS(r.kind) ? `waha się między ${fmt(Math.exp(r.val), 1)} odcieniami` : r.kind === 'vae' ? `błąd na piksel ${fmt(r.val, 4)}` : r.kind === 'diff' ? `błąd zgadywania ${fmt(r.val, 4)}` : r.stats ? `krytyk: ${Math.round(r.stats.pr * 100)}% / ${Math.round(r.stats.pf * 100)}%` : '–';
    [r.label, r.params.toLocaleString('pl'), r.ms ? `${fmt(r.ms, 0)} ms · ${r.device.split(' · ')[0]}` : '–', BITS(r.kind) ? fmt(r.val / LN2) : '–', other].forEach(v => { const td = document.createElement('td'); td.textContent = v; tr.appendChild(td); });
    tb.appendChild(tr);
  });
}

// ---------- start ----------
renderAll(); resetLearn(); renderCmp();
loadData().then(renderGraph);
