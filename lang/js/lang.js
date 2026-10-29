// Laboratorium języka · strona: dane, wybór konstrukcji, wykres, próbki, wizualizacje, porównanie.
const $ = id => document.getElementById(id);
const fmt = (v, d = 2) => v == null || !isFinite(v) ? '–' : v.toFixed(d).replace('.', ',');
const LN2 = Math.log(2);
const show = c => c === ' ' ? '␣' : c === '\n' ? '↵' : c;

// ---------- kroki ----------
const STEPS = [
  { kind: 'bigram', name: '1 · Bigram', sub: 'liczenie par liter', ready: true,
    what: 'Dla każdej litery liczymy, jak często pojawia się po niej każda inna, np. po „r” bardzo często „z”, a po „ż” rzadko spółgłoska. Ta tabela to cała wiedza modelu.',
    why: 'Najprostsza rzecz, która w ogóle przewiduje tekst. To jeszcze nie sieć, tylko liczenie, ale wynik jest punktem odniesienia: każda kolejna konstrukcja musi go pobić.',
    lim: 'Pamięta jedną literę wstecz, więc nie wie, że jest w środku słowa. Pisze polsko brzmiące sylaby bez sensu. Żeby widzieć 3 litery wstecz, tabela urosłaby do ponad 60 mln komórek, a większość z nich nigdy nie pojawiłaby się w tekście.' },
  { kind: 'mlp', name: '2 · Sieć z oknem', sub: 'embeddingi, Bengio 2003', ready: true,
    what: 'Każda litera dostaje wektor kilku liczb (embedding). Sieć bierze kilka ostatnich liter, skleja ich wektory, przepuszcza przez warstwę neuronów, a na końcu ocenia szansę każdej możliwej następnej litery.',
    why: 'Zamiast pamiętać każdą kombinację liter osobno, sieć uczy się podobieństw: litery, które zachowują się podobnie, dostają podobne wektory, więc wiedza o jednej przechodzi na drugą. Na tym pomyśle stoją wszystkie dalsze modele. Zobaczysz to na mapie poniżej.',
    lim: 'Okno ma stałą długość: co wypadło z okna, nie istnieje. Każda pozycja w oknie ma osobne wagi, więc „k” na początku okna i na jego końcu to dla sieci różne rzeczy, a dłuższe okno to szybko więcej parametrów.',
    more: 'Neurony używają funkcji <code>tanh</code>: bez takiej nieliniowości dwie warstwy złożyłyby się w jedno mnożenie macierzy. Optymalizator Adam dobiera wielkość kroku osobno dla każdej wagi.' },
  { kind: 'rnn', name: '3 · Sieć rekurencyjna', sub: 'pamięć: RNN i GRU', ready: true,
    what: 'Sieć czyta tekst litera po literze i niesie ze sobą pamięć: wektor kilkudziesięciu liczb, który po każdej literze aktualizuje. Następną literę przewiduje z tej pamięci, a nie z okna.',
    why: 'Usuwa sztywne okno z kroku 2: te same wagi działają na każdej pozycji, więc kontekst może być dowolnie długi bez dokładania parametrów. Zwykła RNN (Elman 1990) ma jednak problem: przy nauce sygnał o błędzie słabnie z każdą literą wstecz („zanikający gradient”). GRU (2014) dodaje bramki, które same decydują, co w pamięci zostawić, a co nadpisać. Przez zostawioną pamięć gradient przechodzi bez osłabiania.',
    lim: 'Czyta po kolei: litera 50 czeka na 49, więc nauki nie da się zrównoleglić po długości tekstu i trwa ona długo. Cała przeszłość musi się zmieścić w jednym małym wektorze, więc szczegóły sprzed setek liter i tak się zacierają. To naprawia dopiero uwaga (krok 4).',
    more: 'Uczymy na kawałkach po kilkadziesiąt liter (obcięta propagacja wstecz). Gdy gradient nagle urośnie, jest przycinany do długości 1, bo w sieciach rekurencyjnych potrafi eksplodować. Sprawdzanie czyta tekst ciągiem: pierwsze 40 liter rozgrzewa pamięć i nie wlicza się do błędu.' },
  { kind: 'chat', name: '5 · Rozmowa', sub: 'dostrajanie', ready: true, _last: true,
    what: 'Bierze kopię sieci wytrenowanej w kroku 2, 3 albo 4 i uczy ją dalej na krótkich dialogach w formacie „Ty: … / Sieć: …”. Potem w oknie czatu dopisujemy Twoją wiadomość i „Sieć:”, a sieć dokańcza linijkę. To jest cała „rozmowa”: zwykłe przewidywanie dalszego ciągu tekstu w odpowiednim formacie.',
    why: 'Tak samo (w ogromnej skali) powstają asystenci: najpierw sieć uczy się języka z dużej ilości tekstu, potem jest dostrajana na przykładach rozmów. Błąd liczymy tylko na kwestiach Sieci (maskowanie): sieć ma się nauczyć odpowiadać, a nie mówić za rozmówcę. Bez maski uczy się też pisać Twoje kwestie.',
    lim: 'Uczy się formy, nie wiedzy: odpowiada we własnej kolejce, krótkimi zdaniami, w podobnym tonie, ale nie rozumie pytań i nie wie nic poza tym, co przeczytała. 140 rozmów to bardzo mało, więc szybko zaczyna je powtarzać na pamięć (pilnuje tego najlepsza wersja). Sieć z oknem widzi tylko kilka ostatnich znaków, więc nie zobaczy nawet całego pytania.',
    more: 'Dialogi napisał Claude dla tego laboratorium (domena publiczna, plik lang/data/rozmowy). Słownik liter albo tokenów zostaje taki jak w sieci bazowej. Najlepiej dostrajać transformer albo GRU wytrenowane wcześniej na Grabińskim.' },
  { kind: 'transformer', name: '4 · Transformer', sub: 'uwaga, 2017', ready: true,
    what: 'Sieć widzi naraz okno ostatnich liter (np. 64). Każda pozycja zadaje pytanie (zapytanie), każda wcześniejsza ma etykietę (klucz) i treść (wartość). Pozycja bierze treści tych liter, których klucze najlepiej pasują do jej pytania. To jest uwaga. Kilka „głów” pyta równolegle o różne rzeczy, a kilka warstw powtarza to na coraz bardziej przetworzonej treści.',
    why: 'Usuwa wąskie gardło RNN: nic nie musi przecisnąć się przez jeden mały wektor pamięci, bo każda litera może sięgnąć wprost do dowolnej wcześniejszej w oknie, bez osłabiania sygnału. Wszystkie pozycje liczą się jednocześnie, a nie po kolei, więc ogromnie zyskuje na kartach graficznych. To dlatego wygrał i na nim stoją GPT, Claude i inne duże modele.',
    lim: 'Widzi tylko okno: litera sprzed 65 pozycji przy oknie 64 nie istnieje. Koszt uwagi rośnie z kwadratem długości okna (każda pozycja z każdą). Sama uwaga nie zna kolejności liter, trzeba ją dodać osobno (wektory pozycji). Na małych danych i w małej skali nie musi wygrać z GRU: przewaga pojawia się przy większych modelach, tekstach i GPU.',
    more: 'Każdy blok: normalizacja warstwy → uwaga → dodanie do wejścia (połączenie omijające) → normalizacja → mała sieć w każdej pozycji osobno (4× szersza, GELU) → dodanie. Sprawdzanie liczy tylko drugą połowę każdego okna, żeby każda litera miała przed sobą kontekst.' },
];
const HP = {
  chat: [
    { id: 'base', label: 'Sieć bazowa', hint: 'co dostroić', opts: [['transformer', 'transformer (krok 4)'], ['rnn', 'RNN/GRU (krok 3)'], ['mlp', 'sieć z oknem (krok 2)']], val: 'transformer' },
    { id: 'steps', label: 'Kroki', hint: 'dostrajania', opts: [100, 300, 600, 1000], val: 300 },
    { id: 'lr', label: 'Tempo nauki', hint: 'mniejsze niż przy nauce', opts: [.0003, .001, .003], val: .001 },
    { id: 'mask', label: 'Błąd liczony na', hint: 'maskowanie', opts: [[1, 'tylko kwestiach Sieci'], [0, 'całym dialogu']], val: 1 },
    { id: 'device', label: 'Liczy', hint: 'jak liczyć', opts: [['wasm', 'procesor, WASM SIMD'], ['cpu', 'procesor, zwykły JS']], val: 'wasm' },
  ],
  transformer: [
    { id: 'layers', label: 'Warstwy', hint: 'bloków uwagi', opts: [1, 2, 3, 4], val: 2 },
    { id: 'dim', label: 'Szerokość', hint: 'liczb na pozycję', opts: [32, 48, 64], val: 32 },
    { id: 'heads', label: 'Głowy', hint: 'uwagi na warstwę', opts: [1, 2, 4], val: 4 },
    { id: 'ctx', label: 'Okno', hint: 'liter naraz', opts: [32, 64, 128], val: 64 },
    { id: 'steps', label: 'Kroki nauki', hint: 'po 16 okien', opts: [300, 1000, 2000, 4000, 8000], val: 2000 },
    { id: 'lr', label: 'Tempo nauki', hint: 'wielkość kroku', opts: [.001, .003], val: .003 },
    { id: 'device', label: 'Liczy', hint: 'jak liczyć', opts: [['wasm', 'procesor, WASM SIMD'], ['cpu', 'procesor, zwykły JS'], ['gpu', 'karta graficzna (GPU)']], val: 'wasm' },
  ],
  mlp: [
    { id: 'ctx', label: 'Okno', hint: 'liter wstecz', opts: [3, 5, 8, 12], val: 8 },
    { id: 'emb', label: 'Embedding', hint: 'liczb na literę', opts: [2, 8, 16, 32], val: 16 },
    { id: 'hidden', label: 'Neurony', hint: 'w warstwie ukrytej', opts: [64, 128, 256], val: 128 },
    { id: 'steps', label: 'Kroki nauki', hint: 'po 64 przykłady', opts: [1000, 3000, 6000, 12000], val: 3000 },
    { id: 'lr', label: 'Tempo nauki', hint: 'wielkość kroku', opts: [.001, .003, .01], val: .003 },
    { id: 'device', label: 'Liczy', hint: 'jak liczyć', opts: [['wasm', 'procesor, WASM SIMD'], ['cpu', 'procesor, zwykły JS']], val: 'wasm' },
  ],
  rnn: [
    { id: 'cell', label: 'Komórka', hint: 'rodzaj pamięci', opts: [['rnn', 'zwykła RNN'], ['gru', 'GRU (z bramkami)']], val: 'gru' },
    { id: 'hidden', label: 'Pamięć', hint: 'liczb w wektorze', opts: [32, 64, 128], val: 64 },
    { id: 'emb', label: 'Embedding', hint: 'liczb na literę', opts: [8, 16, 32], val: 16 },
    { id: 'seq', label: 'Kawałek', hint: 'liter na raz', opts: [16, 32, 64], val: 32 },
    { id: 'steps', label: 'Kroki nauki', hint: 'po 16 kawałków', opts: [300, 1000, 2000, 4000, 8000], val: 2000 },
    { id: 'lr', label: 'Tempo nauki', hint: 'wielkość kroku', opts: [.001, .003, .01], val: .003 },
    { id: 'device', label: 'Liczy', hint: 'jak liczyć', opts: [['wasm', 'procesor, WASM SIMD'], ['cpu', 'procesor, zwykły JS']], val: 'wasm' },
  ],
};
let cur = 'bigram';

// ---------- dane ----------
let corpus = '', vocab = null, dataLabel = '', unitInfo = null;
// Jednostka tekstu: litery albo tokeny BPE. Błąd sieci na tokenach przeliczamy na literę (dzielimy przez średnią
// liczbę znaków na token), inaczej porównanie byłoby nieuczciwe: przewidzenie całego słowa to więcej niż jednej litery.
const VV = () => unitInfo ? unitInfo.V : vocab ? vocab.chars.length : 2;
const CPT = () => cur === 'chat' && ftInfo ? ftInfo.cpt : unitInfo ? unitInfo.cpt : 1;
let ftInfo = null;
const isBPE = () => !!unitInfo && unitInfo.unit === 'bpe';
const pc = v => v == null ? null : v / CPT();
const showTok = t => t.replace(/ /g, '·').replace(/\n/g, '↵');
const worker = new Worker('js/worker.js');

function loadText(text, label) {
  corpus = cleanText(text);
  vocab = buildVocab(corpus);
  dataLabel = label;
  const nTrain = Math.floor(corpus.length * .9);
  $('dataStats').innerHTML = [
    ['Tekst', label], ['Znaki', corpus.length.toLocaleString('pl')], ['Różne znaki', vocab.chars.length],
    ['Do nauki / sprawdzania', `${nTrain.toLocaleString('pl')} / ${(corpus.length - nTrain).toLocaleString('pl')}`],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd></dd></div>`).join('');
  [...$('dataStats').querySelectorAll('dd')].forEach((dd, i) => dd.textContent = [label, corpus.length.toLocaleString('pl'), vocab.chars.length, `${nTrain.toLocaleString('pl')} / ${(corpus.length - nTrain).toLocaleString('pl')}`][i]);
  $('dataNote').textContent = corpus.length < 200000
    ? `To ${corpus.length.toLocaleString('pl')} znaków. Sieć nauczy się z tego głównie sylab i krótkich słów, a łatwo zapamięta tekst na pamięć. Do sensownej nauki wgraj kilkaset tysięcy znaków.`
    : 'Wystarczająco dużo tekstu, żeby sieć uczyła się języka, a nie tylko zapamiętywała.';
  runs.length = 0; renderCmp();
  sendData();
}
// Wysyła tekst do wątku; przy tokenach wątek najpierw buduje słownik BPE. Tabela porównania zostaje,
// bo błąd na literę da się porównać między jednostkami.
function sendData() {
  if (!corpus) return;
  unitInfo = null; reachDone.length = 0; have.clear(); for (const k in extras) delete extras[k];
  $('train').disabled = true; resetLearn(); renderTokView();
  worker.postMessage({ cmd: 'data', text: corpus, chars: vocab.chars, label: dataLabel, unit: $('unit').value });
}
$('unit').addEventListener('change', sendData);
function renderTokView() {
  const box = $('tokView'), u = unitInfo;
  const st = $('dataStats'), add = (k, v) => { let d = st.querySelector(`[data-k="${k}"]`); if (!d) { d = document.createElement('div'); d.dataset.k = k; d.innerHTML = '<dt></dt><dd></dd>'; st.appendChild(d); } d.querySelector('dt').textContent = k; d.querySelector('dd').textContent = v; };
  st.querySelectorAll('[data-k]').forEach(d => d.remove());
  if (!u || u.unit !== 'bpe') { box.hidden = true; return; }
  add('Tokeny w słowniku', u.V.toLocaleString('pl')); add('Znaków na token', fmt(u.cpt, 2));
  box.hidden = false; box.innerHTML = '';
  const h = (t, cls) => { const e = document.createElement(t); if (cls) e.className = cls; return e; };
  const p1 = h('p', 'note'); p1.textContent = `Ten sam fragment tekstu sprawdzającego podzielony na tokeny (kolejne tokeny mają naprzemienne tło, · to spacja). ${u.n.toLocaleString('pl')} tokenów zamiast ${corpus.length.toLocaleString('pl')} znaków: sieć widzi ${fmt(u.cpt, 1)}× więcej tekstu w tym samym oknie, ale ma ${u.V.toLocaleString('pl')} możliwości do wyboru zamiast ${vocab.chars.length}.`;
  const ex = h('div', 'toks'); u.example.forEach((t, i) => { const s = h('span', i % 2 ? 'b' : 'a'); s.textContent = showTok(t); ex.appendChild(s); });
  const p2 = h('p', 'note'); p2.textContent = 'Pierwsze sklejenia, w kolejności: ' + u.merges.slice(0, 30).map(([a, b]) => `${showTok(a)}+${showTok(b)}`).join('  ');
  const p3 = h('p', 'note'); p3.textContent = 'Najdłuższe tokeny: ' + u.longest.map(showTok).join('  ');
  box.append(p1, ex, p2, p3);
}
// Przykłady: zestawy tekstów z lang/data/ (lista w manifest.json, bo serwer statyczny nie pokazuje zawartości folderu).
// Gdy ich brak albo strona jest otwarta z dysku, zostaje krótki tekst wbudowany w data.js.
// Pliki z Wolnych Lektur kończą się notą licencyjną po linii „-----”; to nie jest tekst utworu.
function stripFooter(t) { const i = t.search(/\r?\n-----\r?\n[^]*$/); return (i > 0 ? t.slice(0, i) : t).replace(/^ISBN[^\n]*\n/m, ''); }
const plural = (n, one, few, many) => n + ' ' + (n === 1 ? one : (n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14)) ? few : many);
let sets = [];
async function loadManifest() {
  try { sets = ((await (await fetch('data/manifest.json', { cache: 'no-cache' })).json()).sets || []).filter(s => s.files && s.files.length); } catch (e) { sets = []; }
  const sel = $('sampleSet'); sel.innerHTML = '';
  sets.forEach(s => sel.add(new Option(s.name, s.id)));
  sel.add(new Option('Krótki tekst Claude\'a (do pokazania uczenia na pamięć)', 'builtin'));
}
async function loadExample() {
  const id = $('sampleSet').value, set = sets.find(s => s.id === id);
  if (!set) { loadText(SAMPLE_TEXT, 'przykład (Claude)'); return; }
  $('useSample').disabled = $('sampleSet').disabled = true;
  try {
    $('dataNote').textContent = 'Wczytuję teksty…';
    const texts = await Promise.all(set.files.map(async f => { const r = await fetch('data/' + set.dir + f.file, { cache: 'force-cache' }); if (!r.ok) throw 0; const t = await r.text(); return set.strip === 'wolnelektury' ? stripFooter(t) : t; }));
    const n = set.files.length, unit = set.strip === 'wolnelektury' ? ['utwór', 'utwory', 'utworów'] : ['artykuł', 'artykuły', 'artykułów'];
    loadText(texts.join('\n\n'), `${set.label}, ${plural(n, ...unit)}`);
    const note = document.createElement('span');
    note.append(`Przykład: ${plural(n, ...unit)} (${set.files.slice(0, 4).map(f => '„' + f.title + '”').join(', ')}${n > 4 ? '…' : ''}). ${set.note} `);
    const a = document.createElement('a'); a.href = 'data/' + set.license; a.target = '_blank'; a.rel = 'noopener'; a.textContent = 'Źródło i licencja.';
    note.append(a, ' ' + $('dataNote').textContent);
    $('dataNote').replaceChildren(note);
  } catch (e) {
    loadText(SAMPLE_TEXT, 'przykład (Claude)');
  } finally { $('useSample').disabled = $('sampleSet').disabled = false; }
}
$('useSample').addEventListener('click', loadExample);
$('files').addEventListener('change', async e => {
  const files = [...e.target.files]; e.target.value = '';
  if (!files.length) return;
  const texts = await Promise.all(files.map(f => f.text()));
  loadText(texts.join('\n\n'), files.length === 1 ? files[0].name : `${files.length} pliki`);
});

// ---------- wybór kroku ----------
function renderSteps() {
  $('steps').innerHTML = '';
  [...STEPS.filter(s => !s._last), ...STEPS.filter(s => s._last)].forEach(s => {
    const b = document.createElement('button');
    b.type = 'button'; b.setAttribute('role', 'tab'); b.id = 'step-' + s.kind;
    b.setAttribute('aria-selected', s.kind === cur ? 'true' : 'false'); b.disabled = !s.ready;
    b.innerHTML = `<span></span><small></small>`; b.firstChild.textContent = s.name; b.lastChild.textContent = s.sub;
    b.addEventListener('click', () => selectStep(s.kind));
    $('steps').appendChild(b);
  });
}
function renderExplain() {
  const s = STEPS.find(x => x.kind === cur);
  $('explain').innerHTML = `<div><h4>Jak działa</h4><p>${s.what}</p></div><div><h4>Dlaczego taka konstrukcja</h4><p>${s.why}</p></div><div><h4 class="lim">Czego nie umie</h4><p>${s.lim}</p></div>` + (s.more ? `<div><h4>Szczegóły</h4><p>${s.more}</p></div>` : '');
}
function renderHP() {
  const list = HP[cur] || [];
  $('hp').innerHTML = '';
  list.forEach(h => {
    const lab = document.createElement('label'); lab.htmlFor = 'hp-' + h.id;
    lab.innerHTML = `<span><b></b><small></small></span>`;
    lab.querySelector('b').textContent = h.label; lab.querySelector('small').textContent = h.hint;
    const sel = document.createElement('select'); sel.id = 'hp-' + h.id;
    h.opts.forEach(o => Array.isArray(o) ? sel.add(new Option(o[1], o[0])) : sel.add(new Option(String(o).replace('.', ','), o)));
    if (h.id === 'device') [...sel.options].forEach(o => {
      if ((o.value === 'gpu' && !('gpu' in navigator)) || (o.value === 'wasm' && !wasmOK)) { o.disabled = true; o.textContent += ' — niedostępne tutaj'; if (h.val === o.value) h.val = 'cpu'; }
    });
    sel.value = h.val; sel.addEventListener('change', () => { h.val = isNaN(+sel.value) ? sel.value : +sel.value; updateParamNote(); if (h.id === 'base') drawViz(); if (h.id === 'device' && h.val === 'gpu') checkGPU(); });
    lab.appendChild(sel); $('hp').appendChild(lab);
  });
  updateParamNote();
}
function hpValues(kind = cur) { const o = {}; (HP[kind] || []).forEach(h => o[h.id] = h.val); o.batch = kind === 'rnn' || kind === 'transformer' ? 16 : 64; return o; }
function paramCount(kind, V, h) {
  if (kind === 'chat') return paramCount(h.base, V, hpValues(h.base));
  if (kind === 'bigram') return V * V;
  if (kind === 'mlp') return V * h.emb + h.ctx * h.emb * h.hidden + h.hidden + h.hidden * V + V;
  if (kind === 'transformer') { const d = h.dim; return V * d + h.ctx * d + h.layers * (12 * d * d + 13 * d) + 2 * d + d * V + V; }
  if (kind === 'rnn') { const k = h.cell === 'gru' ? 3 : 1, H = h.hidden; return V * h.emb + h.emb * k * H + k * H + H * k * H + (h.cell === 'gru' ? H : 0) + H * V + V; }
  return 0;
}
function updateParamNote() {
  if (!vocab) return;
  const n = paramCount(cur, VV(), hpValues());
  $('paramNote').textContent = `${n.toLocaleString('pl')} parametrów` + (cur === 'bigram' ? ' (komórek tabeli)' : ` · ${(n * 4 / 1024).toFixed(0)} KB`);
}

// WebAssembly SIMD: sprawdzenie, czy przeglądarka przyjmie moduł (bez uruchamiania).
const wasmOK = (() => { try { return typeof WebAssembly !== 'undefined' && WebAssembly.validate(Uint8Array.from(atob(WASM_MATMUL_B64), c => c.charCodeAt(0))); } catch (e) { return false; } })();
// ---------- karta graficzna ----------
// Przy pierwszym wyborze GPU: ten sam krok nauki na CPU i na karcie, porównanie wyników.
let gpuState = null;   // null = nie sprawdzano, 'testing', {ok, ...}
function checkGPU() {
  if (gpuState && gpuState.ok) return gpuNote();
  if (gpuState === 'testing') return;
  gpuState = 'testing'; $('train').disabled = true; $('paramNote').textContent = 'Sprawdzam kartę graficzną: ten sam krok nauki na CPU i GPU…';
  worker.postMessage({ cmd: 'gputest' });
}
function gpuNote() {
  if (!gpuState || gpuState === 'testing') return;
  if (gpuState.ok) notice(`Karta: ${gpuState.name}. Test zgodności: ten sam krok nauki na CPU i GPU dał błąd ${fmt(gpuState.lossC, 5)} i ${fmt(gpuState.lossG, 5)}, a wagi po kroku różnią się najwyżej o ${gpuState.diff.toExponential(1).replace('.', ',')} (zaokrąglenia liczb). Na słabej karcie albo przy małej sieci GPU może być wolniejsze od CPU: przesyłanie zadań też kosztuje. Porównaj czas kroku w tabeli.`);
  else notice(`Karta graficzna niedostępna: ${gpuState.msg}. Liczę na procesorze.`);
}
// ---------- nauka ----------
let series = { train: [], val: [] }, live = null, training = false;
const extras = {};                     // ostatnie wizualizacje z wątku, osobno dla każdego kroku
const reachDone = [];                  // zakończone pomiary zasięgu pamięci, do porównania RNN i GRU
const have = new Set();                // kroki, dla których istnieje model (wytrenowany albo wczytany)
let neuronSel = 0, neuronPicked = false, attSel = { head: 0, pos: -1 }, attLayout = null;
function unitLabel() { return isBPE() ? ` · BPE ${unitInfo.V - vocab.chars.length}` : ''; }
function runLabel(kind, hp) {
  if (kind === 'mlp') return `2 · Sieć z oknem (okno ${hp.ctx}, emb ${hp.emb})`;
  if (kind === 'rnn') return `3 · ${hp.cell === 'gru' ? 'GRU' : 'Zwykła RNN'} (pamięć ${hp.hidden})`;
  if (kind === 'chat') return `5 · Rozmowa: ${runLabel(hp.base || 'transformer', HP[hp.base] ? Object.fromEntries(HP[hp.base].map(h => [h.id, h.val])) : {}).replace(/^\d · /, '')}${hp.mask === 0 ? ', bez maski' : ''}`;
  if (kind === 'transformer') return `4 · Transformer (${hp.layers}×${hp.dim}, ${hp.heads} gł., okno ${hp.ctx}${hp.device === 'gpu' ? ', GPU' : ''})`;
  return STEPS.find(s => s.kind === kind).name;
}
function resetLearn() {
  series = { train: [], val: [] }; live = null; $('notice') && $('notice').remove();
  drawChart(); drawViz();
  renderMetrics(null);
  $('sample').textContent = 'Kliknij „Ucz”.';
  $('gen').disabled = !have.has(cur); $('saveModel').disabled = !have.has(cur);
}
$('train').addEventListener('click', () => {
  if (!vocab) return;
  series = { train: [], val: [] }; training = true; delete extras[cur]; neuronPicked = false;
  $('train').disabled = true; $('stop').disabled = cur === 'bigram'; $('saveModel').disabled = true; $('notice') && $('notice').remove();
  $('sample').textContent = cur === 'bigram' ? 'Liczę pary liter…' : 'Sieć zaczyna od losowych wag, więc pierwsze próbki to przypadkowe znaki…';
  if (cur === 'chat') startFinetune(); else worker.postMessage({ cmd: 'train', kind: cur, hp: hpValues() });
  drawViz();
});
let dialogText = null;
async function startFinetune() {
  const hp = hpValues('chat');
  $('sample').textContent = 'Dostrajam: sieć zaczyna od tego, co umie po nauce na tekście bazowym…';
  try {
    if (!dialogText) dialogText = await (await fetch('data/rozmowy/rozmowy.txt')).text();
    worker.postMessage({ cmd: 'finetune', base: hp.base, hp, dialog: dialogText });
  } catch (e) { showErr('Nie udało się wczytać rozmów (data/rozmowy/rozmowy.txt). Strona musi być otwarta przez serwer.'); training = false; $('train').disabled = false; $('stop').disabled = true; }
}
$('stop').addEventListener('click', () => worker.postMessage({ cmd: 'stop' }));
$('temp').addEventListener('input', () => $('tempOut').textContent = (+$('temp').value).toFixed(1).replace('.', ','));
let sampleId = 0;
$('gen').addEventListener('click', () => worker.postMessage({ cmd: 'sample', kind: cur, prompt: $('prompt').value, temp: +$('temp').value, len: 400, id: ++sampleId }));
$('saveModel').addEventListener('click', () => worker.postMessage({ cmd: 'export', kind: cur }));
$('loadModel').addEventListener('change', async e => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try { worker.postMessage({ cmd: 'import', json: JSON.parse(await f.text()) }); }
  catch (err) { showErr('To nie jest poprawny plik JSON.'); }
});
function showErr(msg) { $('err').textContent = 'Błąd: ' + msg; $('err').hidden = false; }
function notice(text) {
  let n = $('notice'); if (!n) { n = document.createElement('p'); n.id = 'notice'; n.className = 'notice'; $('metrics').parentNode.after(n); }
  n.textContent = text;
}
function selectStep(kind) { cur = kind; renderSteps(); renderExplain(); renderHP(); resetLearn(); renderChat(); }

worker.onmessage = e => {
  const m = e.data;
  if (m.type === 'error') { showErr(m.msg); training = false; $('train').disabled = false; $('stop').disabled = true; chatBusy(false); return; }
  if (m.type === 'ftinfo') { ftInfo = m; notice(`Rozmowy: ${m.conv} dialogów (${m.conv - m.val} do nauki, ${m.val} do sprawdzania), ${m.tokens.toLocaleString('pl')} jednostek tekstu, z czego ${m.learn.toLocaleString('pl')} to kwestie Sieci.`); return; }
  if (m.type === 'chat') { chatAnswer(m); return; }
  if (m.type === 'tokprogress') { $('paramNote').textContent = `Buduję słownik tokenów: ${m.r} / ${m.n} sklejeń…`; return; }
  if (m.type === 'ready') { unitInfo = m; $('train').disabled = training; renderTokView(); updateParamNote(); drawChart(); renderMetrics(null); return; }
  if (m.type === 'gputest') {
    gpuState = m; $('train').disabled = training;
    if (!m.ok) { const h = HP.transformer.find(x => x.id === 'device'); h.val = 'cpu'; if ($('hp-device')) $('hp-device').value = 'cpu'; }
    updateParamNote(); gpuNote(); return;
  }
  if (m.type === 'sample') { if (m.text != null) $('sample').textContent = (isBPE() ? $('prompt').value.replace(/[^\S\n]+$/, '') : $('prompt').value) + m.text; return; }
  if (m.type === 'export') {
    const blob = new Blob([JSON.stringify(m.json)], { type: 'application/json' }), a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = `neurolab-lang-${m.kind}${m.json.hp && m.json.hp.cell ? '-' + m.json.hp.cell : ''}-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    return;
  }
  if (m.type === 'imported') {
    have.add(m.kind);
    if (m.hp) (HP[m.kind] || []).forEach(h => { if (m.hp[h.id] != null) h.val = m.hp[h.id]; });
    selectStep(m.kind);
    $('sample').textContent = m.sample;
    renderMetrics({ step: m.step || 0, val: m.val, params: m.params });
    notice(m.same
      ? `Wczytany model (${runLabel(m.kind, m.hp || {})}), uczony na: ${m.trainedOn || 'nieznany tekst'}. Błąd sprawdzający policzony na obecnym tekście.`
      : `Wczytany model ma inny słownik liter niż obecny tekst (uczony na: ${m.trainedOn || 'nieznany tekst'}), więc nie da się go sprawdzić na tym tekście. Generowanie działa.`);
    if (m.same && m.val != null) { runs.push({ kind: m.kind, label: runLabel(m.kind, m.hp || {}) + unitLabel() + ' · wczytany', params: m.params, val: pc(m.val), sample: m.sample }); renderCmp(); }
    return;
  }
  if (m.type !== 'progress' && m.type !== 'done') return;
  if (m.kind !== cur) return;
  if (m.train != null) series.train.push([m.step, m.train]);
  if (m.val != null && (!series.val.length || series.val[series.val.length - 1][0] !== m.step)) series.val.push([m.step, m.val]);
  live = m;
  if (m.heat || m.emb || m.reach || m.neurons || m.attn) { extras[m.kind] = { ...(extras[m.kind] || {}), ...pickExtras(m) }; drawViz(); }
  drawChart(); renderMetrics(m);
  if (m.sample) $('sample').textContent = m.sample;
  if (m.type === 'done') {
    training = false; have.add(m.kind);
    $('train').disabled = false; $('stop').disabled = true; $('gen').disabled = false; $('saveModel').disabled = false;
    if (m.restored) notice(`Pod koniec błąd sprawdzający zaczął rosnąć, więc zostawiłem wersję sieci z kroku ${m.bestStep.toLocaleString('pl')} (najlepszą). Tak działa „wczesne zatrzymanie”.`);
    if (m.reach) { reachDone.push({ label: runLabel(m.kind, m.hp), r: m.reach }); if (reachDone.length > 4) reachDone.shift(); drawViz(); }
    runs.push({ kind: m.kind, label: runLabel(m.kind, m.hp || hpValues(m.kind)) + unitLabel(), params: m.params, val: pc(m.val), sample: m.sample, stopped: m.stopped, ms: m.ms, device: m.device });
    renderCmp();
  }
};
function pickExtras(m) { const o = {}; ['heat', 'emb', 'reach', 'neurons', 'attn'].forEach(k => { if (m[k]) o[k] = m[k]; }); if (m.hp) o.hp = m.hp; return o; }

function renderMetrics(m) {
  const V = VV(), val = m ? pc(m.val) : null, bpe = isBPE();
  const best = m && m.bestStep != null && m.step > 1 ? `najlepsza wersja: krok ${m.bestStep.toLocaleString('pl')}` : '';
  const rows = [
    ['Błąd na literę', fmt(val, 3), (m && m.train != null ? `treningowy ${fmt(pc(m.train), 3)}` : 'sprawdzający; mniej = lepiej') + (bpe && m && m.val != null ? ` · na token ${fmt(m.val, 3)}` : '')],
    ['Bity na znak', fmt(val != null ? val / LN2 : null), `bez wiedzy: ${fmt(Math.log(V) / CPT() / LN2)}`],
    ['Waha się między', val != null ? `${fmt(Math.exp(val), 1)} literami` : '–', bpe ? (m && m.val != null ? `albo ${fmt(Math.exp(m.val), 0)} tokenami z ${V.toLocaleString('pl')}` : `${V.toLocaleString('pl')} tokenów w słowniku`) : `z ${V} możliwych`],
    ['Krok', m ? m.step.toLocaleString('pl') : '–', m ? `${m.params.toLocaleString('pl')} parametrów` + (m.ms ? ` · ${fmt(m.ms, 0)} ms/krok (${m.device})` : '') + (best ? ' · ' + best : '') : ''],
  ];
  $('metrics').innerHTML = rows.map(() => '<div><dt></dt><dd></dd></div>').join('');
  [...$('metrics').children].forEach((d, i) => { d.querySelector('dt').textContent = rows[i][0]; const dd = d.querySelector('dd'); dd.textContent = rows[i][1]; const s = document.createElement('small'); s.textContent = rows[i][2]; dd.appendChild(s); });
  if (m && m.train != null && m.val != null && m.val - m.train > .4) {
    const d = document.createElement('div'); d.className = 'overfit';
    d.innerHTML = '<dt>Uczy się na pamięć</dt><dd></dd>';
    d.querySelector('dd').textContent = 'Na tekście treningowym błąd spada, a na sprawdzającym rośnie: sieć zapamiętuje konkretne zdania zamiast reguł języka. Pomaga więcej tekstu, mniejsza sieć albo mniej kroków nauki.';
    $('metrics').appendChild(d);
  }
}

// ---------- wykres błędu ----------
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function prepCanvas(cv, w, h) { const d = window.devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const c = cv.getContext('2d'); c.setTransform(d, 0, 0, d, 0, 0); return c; }
function drawChart() {
  const W = 640, H = 260, c = prepCanvas($('chart'), W, H), L = 46, R = 12, T = 14, B = 30;
  c.clearRect(0, 0, W, H);
  const V = VV(), k = CPT(), sT = series.train.map(([a, b]) => [a, b / k]), sV = series.val.map(([a, b]) => [a, b / k]), pts = [...sT, ...sV];
  const top = Math.max(Math.log(V) / k, ...pts.map(p => p[1])) * 1.05, maxX = Math.max(1, ...pts.map(p => p[0]));
  const lo = Math.max(0, Math.min(top * .3, ...pts.map(p => p[1])) - .2);
  const x = s => L + (s / maxX) * (W - L - R), y = v => T + (1 - (v - lo) / (top - lo)) * (H - T - B);
  c.font = '11px "Atkinson Hyperlegible",system-ui,sans-serif'; c.fillStyle = css('--muted'); c.strokeStyle = css('--line'); c.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const v = lo + (top - lo) * k / 4, yy = y(v); c.beginPath(); c.moveTo(L, yy); c.lineTo(W - R, yy); c.stroke(); c.textAlign = 'right'; c.textBaseline = 'middle'; c.fillText(fmt(v, 1), L - 6, yy); }
  c.textAlign = 'center'; c.textBaseline = 'top';
  [0, .5, 1].forEach(f => c.fillText(Math.round(maxX * f).toLocaleString('pl'), x(maxX * f), H - B + 8));
  // poziom „bez wiedzy”: każda litera równie prawdopodobna
  const yb = y(Math.log(V) / k); c.setLineDash([4, 4]); c.strokeStyle = css('--muted'); c.beginPath(); c.moveTo(L, yb); c.lineTo(W - R, yb); c.stroke(); c.setLineDash([]);
  c.textAlign = 'left'; c.textBaseline = 'bottom'; c.fillText('bez wiedzy (losowe litery)', L + 6, yb - 3);
  const line = (arr, col, w) => { if (arr.length < 2) { if (arr.length === 1) { c.fillStyle = col; c.beginPath(); c.arc(x(arr[0][0]), y(arr[0][1]), 3, 0, 7); c.fill(); } return; } c.strokeStyle = col; c.lineWidth = w; c.beginPath(); arr.forEach((p, i) => i ? c.lineTo(x(p[0]), y(p[1])) : c.moveTo(x(p[0]), y(p[1]))); c.stroke(); };
  line(sT, css('--muted'), 1.4); line(sV, css('--rot'), 2.2);
  // najlepszy punkt: tu zostanie zapamiętana wersja sieci
  if (live && live.bestVal != null && live.step > 1 && series.val.length > 1) {
    const bx = x(live.bestStep), by = y(live.bestVal / k);
    c.strokeStyle = css('--fg'); c.lineWidth = 1.5; c.beginPath(); c.arc(bx, by, 5, 0, 7); c.stroke();
    c.fillStyle = css('--fg'); c.textAlign = bx > W - 120 ? 'right' : 'left'; c.textBaseline = 'top'; c.fillText('najlepsza', bx + (bx > W - 120 ? -8 : 8), by + 4);
  }
  if (!pts.length) { c.fillStyle = css('--muted'); c.textAlign = 'center'; c.textBaseline = 'middle'; c.font = '13px "Atkinson Hyperlegible",system-ui,sans-serif'; c.fillText('Tu pojawi się krzywa błędu.', W / 2, H / 2); }
}

// ---------- wizualizacje ----------
const VIZ_MODES = { rnn: [['neurons', 'Neurony pamięci'], ['reach', 'Zasięg pamięci'], ['emb', 'Mapa liter']], transformer: [['attn', 'Uwaga'], ['emb', 'Mapa liter']] };
const vizModeOf = {};
const vizKind = () => cur === 'chat' ? hpValues('chat').base : cur;
function curMode() { const list = VIZ_MODES[vizKind()]; if (!list) return 'emb'; if (!list.some(x => x[0] === vizModeOf[cur])) vizModeOf[cur] = list[0][0]; return vizModeOf[cur]; }
function renderVizTabs() {
  const box = $('vizTabs'), list = VIZ_MODES[vizKind()];
  box.hidden = !list; if (!list) return;
  box.innerHTML = ''; const vizMode = curMode();
  list.forEach(([id, name]) => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = name; b.setAttribute('aria-pressed', vizMode === id ? 'true' : 'false');
    b.addEventListener('click', () => { vizModeOf[cur] = id; drawViz(); }); box.appendChild(b);
  });
}
function waitViz(c, W, H, text) { c.fillStyle = css('--muted'); c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(text, W / 2, H / 2); }
function drawViz() {
  const W = 640, H = 420, c = prepCanvas($('viz'), W, H), ex = extras[cur] || {};
  c.clearRect(0, 0, W, H);
  c.font = '13px "Atkinson Hyperlegible",system-ui,sans-serif';
  renderVizTabs();
  $('vizCtl').hidden = true;
  if (vizKind() === 'bigram') {
    $('vizTitle').textContent = 'Tabela par liter';
    if (!ex.heat) { $('vizNote').textContent = 'Po policzeniu zobaczysz, po jakich literach co zwykle następuje.'; return waitViz(c, W, H, 'Czeka na naukę.'); }
    $('vizNote').textContent = 'Wiersz: poprzednia litera. Kolumna: następna. Im jaśniej, tym częściej. Pokazane najczęstsze znaki; ␣ to spacja, ↵ to nowa linia.';
    const { labels, m: M } = ex.heat, n = labels.length, pad = 26, cell = Math.min((W - pad - 6) / n, (H - pad - 6) / n);
    c.textAlign = 'center'; c.textBaseline = 'middle'; c.font = '11px "JetBrains Mono",monospace'; c.fillStyle = css('--muted');
    const lab = l => { const t = showTok(l); return t.length > 3 ? t.slice(0, 3) : t; };
    if (isBPE()) c.font = '9px "JetBrains Mono",monospace';
    labels.forEach((l, i) => { c.fillText(lab(l), pad + cell * i + cell / 2, pad / 2); c.fillText(lab(l), pad / 2, pad + cell * i + cell / 2); });
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      const v = Math.sqrt(M[a][b]);       // pierwiastek, żeby rzadkie pary też było widać
      c.fillStyle = `rgba(156,172,112,${Math.min(1, v * 1.6).toFixed(3)})`;
      c.fillRect(pad + cell * b + .5, pad + cell * a + .5, cell - 1, cell - 1);
    }
    return;
  }
  const mode = curMode(); attLayout = null;
  if (mode === 'attn') return drawAttention(c, W, H, ex.attn);
  if (mode === 'neurons') return drawNeurons(c, W, H, ex.neurons);
  if (mode === 'reach') return drawReach(c, W, H, ex.reach);
  $('vizTitle').textContent = 'Mapa liter (embeddingi)';
  if (!ex.emb) { $('vizNote').textContent = 'Po chwili nauki zobaczysz, jak sieć rozmieszcza litery: podobne zachowanie = blisko siebie.'; return waitViz(c, W, H, 'Czeka na naukę.'); }
  $('vizNote').textContent = 'Każda litera to punkt; sieć sama ustawia je tak, żeby podobnie zachowujące się litery leżały blisko. Zielone: samogłoski, białe: spółgłoski, niebieskie: odstępy i interpunkcja. Rzut wielu wymiarów na płaszczyznę (PCA), więc odległości są przybliżone.';
  const { labels, pts } = ex.emb, xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), P = 26;
  const X = v => P + (v - x0) / ((x1 - x0) || 1) * (W - 2 * P), Y = v => P + (v - y0) / ((y1 - y0) || 1) * (H - 2 * P);
  c.textAlign = 'center'; c.textBaseline = 'middle'; c.font = isBPE() ? '11px "JetBrains Mono",monospace' : '15px "JetBrains Mono",monospace';
  labels.forEach((l, i) => {
    const t = l.trim(), vow = t.length > 0 && POLISH_VOWELS.has(t[t.length - 1]), letter = /\p{L}/u.test(l);
    c.fillStyle = vow ? css('--rot') : letter ? css('--fg') : '#7fa6d8';
    c.fillText(showTok(l), X(pts[i][0]), Y(pts[i][1]));
  });
}

// Neurony pamięci: tło każdej litery pokazuje, co dany neuron „myśli” po jej przeczytaniu.
function drawNeurons(c, W, H, nr) {
  $('vizTitle').textContent = 'Neurony pamięci';
  if (!nr) { $('vizNote').textContent = 'Po chwili nauki zobaczysz, na co reagują pojedyncze liczby w pamięci sieci.'; return waitViz(c, W, H, 'Czeka na naukę.'); }
  if (!neuronPicked && nr.found.length) neuronSel = nr.found[0].neuron;
  neuronSel = Math.min(neuronSel, nr.H - 1);
  const ctl = $('vizCtl'); ctl.hidden = false; ctl.innerHTML = '';
  nr.found.forEach(f => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip';
    b.textContent = `${f.name}: neuron ${f.neuron} (r = ${fmt(f.r, 2)})`; b.setAttribute('aria-pressed', f.neuron === neuronSel ? 'true' : 'false');
    b.addEventListener('click', () => { neuronSel = f.neuron; neuronPicked = true; drawViz(); }); ctl.appendChild(b);
  });
  const prev = document.createElement('button'), next = document.createElement('button'), out = document.createElement('output');
  prev.type = next.type = 'button'; prev.className = next.className = 'ghost nav'; prev.textContent = '◀'; next.textContent = '▶';
  prev.setAttribute('aria-label', 'Poprzedni neuron'); next.setAttribute('aria-label', 'Następny neuron');
  out.textContent = `neuron ${neuronSel} / ${nr.H - 1}`;
  prev.addEventListener('click', () => { neuronSel = (neuronSel + nr.H - 1) % nr.H; neuronPicked = true; drawViz(); });
  next.addEventListener('click', () => { neuronSel = (neuronSel + 1) % nr.H; neuronPicked = true; drawViz(); });
  ctl.append(prev, out, next);
  $('vizNote').textContent = 'Tekst sprawdzający, a pod każdą literą stan wybranego neuronu po jej przeczytaniu: zielony = dodatni, niebieski = ujemny, im mocniej, tym bliżej ±1. Przyciski u góry to neurony, które najmocniej pokrywają się z prostą cechą tekstu (r to współczynnik korelacji). Nikt tych cech sieci nie podawał: sama uznała je za przydatne do przewidywania.';
  const ch = 24, cells = layoutPieces(nr.pieces, W, ch), rowsMax = Math.floor((H - 8) / ch), last = cells.length ? cells[cells.length - 1].row : 0, off = Math.max(0, last - rowsMax + 1);
  c.font = '15px "JetBrains Mono",monospace'; c.textAlign = 'center'; c.textBaseline = 'middle';
  cells.forEach(({ x, row, w, t }, i) => {
    if (row < off) return;
    const a = Math.max(-1, Math.min(1, nr.acts[i * nr.H + neuronSel])), y = 4 + (row - off) * ch;
    c.fillStyle = a >= 0 ? `rgba(156,172,112,${(a * .9).toFixed(3)})` : `rgba(127,166,216,${(-a * .9).toFixed(3)})`;
    c.fillRect(x, y, w - 1, ch - 2);
    c.fillStyle = Math.abs(a) > .6 ? css('--bg') : css('--fg');
    c.fillText(t, x + w / 2, y + ch / 2);
  });
}
// Układa kawałki tekstu (litery albo tokeny) w wiersze: szerokość kawałka zależy od liczby znaków.
function layoutPieces(pieces, W, ch) {
  const cw = 17, out = []; let x = 8, row = 0;
  pieces.forEach(p => {
    const t = showTok(p), w = Math.max(1, [...t].length) * cw;
    if (x + w > W - 8 && x > 8) { x = 8; row++; }
    out.push({ x, row, w, t });
    x += w + (isBPE() ? 2 : 0);
    if (p.includes('\n')) { x = 8; row++; }
  });
  return out;
}

// Uwaga: wybierz literę (klik), a tło wcześniejszych liter pokaże, ile uwagi od niej dostają w wybranej głowie.
function drawAttention(c, W, H, at) {
  $('vizTitle').textContent = 'Uwaga';
  if (!at) { $('vizNote').textContent = 'Po chwili nauki zobaczysz, na które wcześniejsze litery patrzy każda pozycja.'; return waitViz(c, W, H, 'Czeka na naukę.'); }
  const nH = at.L * at.nh; attSel.head = Math.min(attSel.head, nH - 1);
  if (attSel.pos < 0 || attSel.pos >= at.T) attSel.pos = at.T - 1;
  const ctl = $('vizCtl'); ctl.hidden = false; ctl.innerHTML = '';
  at.heads.forEach((h, k) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'chip';
    b.textContent = `warstwa ${h.layer + 1}, głowa ${h.head + 1}: średnio ${fmt(h.dist, 1)} ${isBPE() ? 'tok.' : 'lit.'} wstecz`;
    b.setAttribute('aria-pressed', k === attSel.head ? 'true' : 'false');
    b.addEventListener('click', () => { attSel.head = k; drawViz(); }); ctl.appendChild(b);
  });
  $('vizNote').textContent = 'Kliknij literę: obramowana to pozycja, która „pyta”, a tło liter przed nią pokazuje, ile uwagi każda z nich dostaje (najjaśniejsza = najwięcej, razem 100%). Litery po prawej są szare: sieć nie może patrzeć w przyszłość. Głowy uczą się różnych zadań: jedne patrzą na poprzednią literę, inne na początek słowa albo na spację. Liczba przy głowie mówi, jak daleko średnio sięga.';
  const { layer, head } = at.heads[attSel.head], A = at.att[layer], T = at.T, i = attSel.pos, row_ = A.subarray(head * T * T + i * T, head * T * T + i * T + T);
  let mx = 0; for (let j = 0; j <= i; j++) mx = Math.max(mx, row_[j]);
  const txt = at.pieces, ch = 30, cells = layoutPieces(txt, W, ch);
  c.font = '15px "JetBrains Mono",monospace'; c.textAlign = 'center'; c.textBaseline = 'middle';
  let r = 0;
  cells.forEach(({ x, row, w, t }, j) => {
    const y = 8 + row * ch; r = row;
    if (j <= i) { const a = row_[j] / (mx || 1); c.fillStyle = `rgba(156,172,112,${(a * .95).toFixed(3)})`; c.fillRect(x, y, w - 1, ch - 6); }
    if (j === i) { c.strokeStyle = css('--fg'); c.lineWidth = 2; c.strokeRect(x + 1, y + 1, w - 3, ch - 8); }
    c.fillStyle = j > i ? 'rgba(150,150,165,.4)' : (row_[j] / (mx || 1) > .6 ? css('--bg') : css('--fg'));
    c.fillText(t, x + w / 2, y + (ch - 6) / 2);
  });
  attLayout = cells.map(({ x, row, w }) => [x, 8 + row * ch, w, ch - 4]);
  // najmocniejsze trzy litery
  const top = [...Array(i + 1).keys()].sort((a, b) => row_[b] - row_[a]).slice(0, 3);
  c.font = '13px "Atkinson Hyperlegible",system-ui,sans-serif'; c.textAlign = 'left'; c.fillStyle = css('--muted');
  const show2 = t => t === ' ' ? 'spacja' : t === '\n' ? 'nowa linia' : `„${showTok(t)}”`;
  const y0 = 8 + (r + 1) * ch + 14;
  c.fillText(`Pozycja ${i + 1} (${show2(txt[i])}) patrzy najmocniej na:`, 8, y0);
  c.fillText(top.map(j => `${show2(txt[j])} ${i - j === 0 ? '(siebie)' : `(${i - j} wstecz)`} ${Math.round(row_[j] * 100)}%`).join(',  '), 8, y0 + 20);
}
$('viz').addEventListener('click', e => {
  if (!attLayout) return;
  const rc = e.currentTarget.getBoundingClientRect(), x = (e.clientX - rc.left) * 640 / rc.width, y = (e.clientY - rc.top) * 420 / rc.height;
  const k = attLayout.findIndex(([cx, cy, w, h]) => x >= cx && x < cx + w && y >= cy && y < cy + h);
  if (k >= 0) { attSel.pos = k; drawViz(); }
});

// Zasięg pamięci: jak silny sygnał uczący dociera do pamięci sprzed d liter (skala logarytmiczna).
function drawReach(c, W, H, reach) {
  $('vizTitle').textContent = 'Zasięg pamięci';
  $('vizNote').textContent = 'Błąd liczony tylko na ostatniej literze. Linia pokazuje, jaka część sygnału uczącego (gradientu) dociera do pamięci sprzed 1, 2, … 59 liter. Spadek o jedną kreskę to dziesięć razy słabiej. Gdzie sygnał zanika, tam sieć nie ma jak nauczyć się zależności. Zwykła RNN gubi go zwykle po kilkunastu literach, GRU niesie dalej. Przerywane linie to wcześniejsze zakończone nauki.';
  if (!reach && !reachDone.length) return waitViz(c, W, H, 'Czeka na naukę.');
  const L = 54, R = 14, T = 14, B = 34, lo = -8, hi = 0.5, n = 60;
  const x = d => L + d / (n - 1) * (W - L - R), y = v => T + (hi - Math.max(lo, Math.log10(Math.max(v, 1e-12)))) / (hi - lo) * (H - T - B);
  c.font = '11px "Atkinson Hyperlegible",system-ui,sans-serif'; c.strokeStyle = css('--line'); c.fillStyle = css('--muted'); c.lineWidth = 1;
  for (let e = 0; e >= lo; e -= 2) { const yy = y(10 ** e); c.beginPath(); c.moveTo(L, yy); c.lineTo(W - R, yy); c.stroke(); c.textAlign = 'right'; c.textBaseline = 'middle'; c.fillText(e === 0 ? '100%' : `10^${e}`, L - 6, yy); }
  c.textAlign = 'center'; c.textBaseline = 'top';
  [0, 10, 20, 30, 40, 50, 59].forEach(d => c.fillText(d, x(d), H - B + 6));
  c.fillText('liter wstecz', (L + W - R) / 2, H - 14);
  const colors = ['#7fa6d8', '#d0605e', '#c9a86a', '#b48ad0'];
  const drawLine = (r, col, dash, w) => { c.strokeStyle = col; c.lineWidth = w; c.setLineDash(dash); c.beginPath(); r.slice(0, n).forEach((v, d) => d ? c.lineTo(x(d), y(v)) : c.moveTo(x(d), y(v))); c.stroke(); c.setLineDash([]); };
  reachDone.forEach((rd, i) => drawLine(rd.r, colors[i % colors.length], [5, 4], 1.5));
  if (reach && training) drawLine(reach, css('--rot'), [], 2.4);
  // legenda
  c.textAlign = 'left'; c.textBaseline = 'middle'; let ly = T + 8;
  const leg = (col, text, dash) => { c.strokeStyle = col; c.lineWidth = 2; c.setLineDash(dash); c.beginPath(); c.moveTo(W - R - 210, ly); c.lineTo(W - R - 186, ly); c.stroke(); c.setLineDash([]); c.fillStyle = css('--fg'); c.fillText(text, W - R - 180, ly); ly += 16; };
  if (reach && training) leg(css('--rot'), 'trwająca nauka', []);
  reachDone.forEach((rd, i) => leg(colors[i % colors.length], rd.label.replace(/^3 · /, ''), [5, 4]));
}

// ---------- okno rozmowy ----------
const chatHist = [];
let chatId = 0;
function renderChat() {
  $('chatBox').hidden = cur !== 'chat';
  const log = $('chatLog'); log.innerHTML = '';
  if (!chatHist.length) { const p = document.createElement('p'); p.className = 'note'; p.textContent = 'Napisz coś krótkiego, np. „Cześć, jak się masz?”. Możesz rozmawiać z siecią przed dostrojeniem i po nim: różnica pokazuje, czego uczy dostrajanie.'; log.appendChild(p); }
  chatHist.forEach(t => { const d = document.createElement('div'); d.className = 'msg ' + (t.who === 'Ty' ? 'me' : 'net'); const b = document.createElement('b'); b.textContent = t.who === 'Ty' ? 'Ty' : `Sieć (${t.from})`; d.append(b, ' ', t.text || '…'); log.appendChild(d); });
  log.scrollTop = log.scrollHeight;
}
function chatBusy(on) { $('chatSend').disabled = on; $('chatMsg').disabled = on; }
function sendChat() {
  const msg = $('chatMsg').value.trim(); if (!msg) return;
  const who = $('chatWho').value, base = hpValues('chat').base, slot = who === 'chat' ? 'chat' : base;
  if (!have.has(slot)) { showErr(who === 'chat' ? 'Najpierw dostrój sieć (przycisk „Ucz” w kroku 5).' : `Najpierw naucz sieć bazową (${runLabel(base, hpValues(base))}).`); return; }
  $('err').hidden = true;
  chatHist.push({ who: 'Ty', text: msg }); $('chatMsg').value = '';
  // historia: ostatnie kwestie w formacie z nauki, na końcu „Sieć:”, które sieć dokańcza
  const hist = chatHist.slice(-8).map(t => `${t.who === 'Ty' ? 'Ty' : 'Sieć'}: ${t.text}`).join('\n') + '\nSieć:';
  chatHist.push({ who: 'Sieć', text: '', from: who === 'chat' ? 'dostrojona' : 'przed dostrojeniem', id: ++chatId });
  renderChat(); chatBusy(true);
  worker.postMessage({ cmd: 'chat', slot, history: hist, temp: +$('temp').value, id: chatId });
}
function chatAnswer(m) {
  const t = chatHist.find(x => x.id === m.id); if (t) t.text = m.text || '(nic)';
  chatBusy(false); renderChat(); $('chatMsg').focus();
}
$('chatSend').addEventListener('click', sendChat);
$('chatMsg').addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });
$('chatClear').addEventListener('click', () => { chatHist.length = 0; renderChat(); });

// ---------- porównanie ----------
const runs = [];
function renderCmp() {
  const tb = $('cmp').querySelector('tbody'); tb.innerHTML = '';
  if (!runs.length) { tb.innerHTML = '<tr><td colspan="7" class="note">Tu trafi każda zakończona nauka, żeby porównać konstrukcje na tym samym tekście.</td></tr>'; return; }
  const best = Math.min(...runs.map(r => r.val));
  runs.forEach(r => {
    const tr = document.createElement('tr'); if (r.val === best) tr.className = 'best';
    [r.label + (r.stopped ? ' · przerwana' : ''), r.params.toLocaleString('pl'), r.ms ? `${fmt(r.ms, 0)} ms · ${r.device.split(' · ')[0]}` : '–', fmt(r.val, 3), fmt(r.val / LN2), fmt(Math.exp(r.val), 1), r.sample.slice(0, 60)].forEach((v, i) => {
      const td = document.createElement('td'); td.textContent = v; if (i === 6) td.className = 's'; tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
}

// ---------- start ----------
renderSteps(); renderExplain(); renderHP(); renderCmp(); renderChat();
loadManifest().then(loadExample);
