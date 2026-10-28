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
  { kind: 'rnn', name: '3 · Sieć rekurencyjna', sub: 'pamięć, wkrótce', ready: false },
  { kind: 'transformer', name: '4 · Transformer', sub: 'uwaga, wkrótce', ready: false },
];
const HP = {
  mlp: [
    { id: 'ctx', label: 'Okno', hint: 'liter wstecz', opts: [3, 5, 8, 12], val: 8 },
    { id: 'emb', label: 'Embedding', hint: 'liczb na literę', opts: [2, 8, 16, 32], val: 16 },
    { id: 'hidden', label: 'Neurony', hint: 'w warstwie ukrytej', opts: [64, 128, 256], val: 128 },
    { id: 'steps', label: 'Kroki nauki', hint: 'po 64 przykłady', opts: [1000, 3000, 6000, 12000], val: 3000 },
    { id: 'lr', label: 'Tempo nauki', hint: 'wielkość kroku', opts: [.001, .003, .01], val: .003 },
  ],
};
let cur = 'bigram';

// ---------- dane ----------
let corpus = '', vocab = null, dataLabel = '';
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
  runs.length = 0; renderCmp(); resetLearn();
  worker.postMessage({ cmd: 'data', text: corpus, chars: vocab.chars });
  updateParamNote();
}
$('useSample').addEventListener('click', () => loadText(SAMPLE_TEXT, 'przykład (Claude)'));
$('files').addEventListener('change', async e => {
  const files = [...e.target.files]; e.target.value = '';
  if (!files.length) return;
  const texts = await Promise.all(files.map(f => f.text()));
  loadText(texts.join('\n\n'), files.length === 1 ? files[0].name : `${files.length} pliki`);
});

// ---------- wybór kroku ----------
function renderSteps() {
  $('steps').innerHTML = '';
  STEPS.forEach(s => {
    const b = document.createElement('button');
    b.type = 'button'; b.setAttribute('role', 'tab'); b.id = 'step-' + s.kind;
    b.setAttribute('aria-selected', s.kind === cur ? 'true' : 'false'); b.disabled = !s.ready;
    b.innerHTML = `<span></span><small></small>`; b.firstChild.textContent = s.name; b.lastChild.textContent = s.sub;
    b.addEventListener('click', () => { cur = s.kind; renderSteps(); renderExplain(); renderHP(); resetLearn(); });
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
    h.opts.forEach(o => sel.add(new Option(String(o).replace('.', ','), o)));
    sel.value = h.val; sel.addEventListener('change', () => { h.val = +sel.value; updateParamNote(); });
    lab.appendChild(sel); $('hp').appendChild(lab);
  });
  updateParamNote();
}
function hpValues() { const o = {}; (HP[cur] || []).forEach(h => o[h.id] = h.val); o.batch = 64; return o; }
function paramCount(kind, V, h) {
  if (kind === 'bigram') return V * V;
  if (kind === 'mlp') return V * h.emb + h.ctx * h.emb * h.hidden + h.hidden + h.hidden * V + V;
  return 0;
}
function updateParamNote() {
  if (!vocab) return;
  const n = paramCount(cur, vocab.chars.length, hpValues());
  $('paramNote').textContent = `${n.toLocaleString('pl')} parametrów` + (cur === 'mlp' ? ` · ${(n * 4 / 1024).toFixed(0)} KB` : ' (komórek tabeli)');
}

// ---------- nauka ----------
let series = { train: [], val: [] }, live = null, training = false;
function resetLearn() {
  series = { train: [], val: [] }; live = null;
  drawChart(); drawViz(null);
  $('metrics').innerHTML = ''; renderMetrics(null);
  $('sample').textContent = 'Kliknij „Ucz”.';
  $('gen').disabled = !runs.some(r => r.kind === cur);
}
$('train').addEventListener('click', () => {
  if (!vocab) return;
  series = { train: [], val: [] }; training = true;
  $('train').disabled = true; $('stop').disabled = cur === 'bigram';
  $('sample').textContent = cur === 'bigram' ? 'Liczę pary liter…' : 'Sieć zaczyna od losowych wag, więc pierwsze próbki to przypadkowe znaki…';
  worker.postMessage({ cmd: 'train', kind: cur, hp: hpValues() });
});
$('stop').addEventListener('click', () => worker.postMessage({ cmd: 'stop' }));
$('temp').addEventListener('input', () => $('tempOut').textContent = (+$('temp').value).toFixed(1).replace('.', ','));
let sampleId = 0;
$('gen').addEventListener('click', () => worker.postMessage({ cmd: 'sample', kind: cur, prompt: $('prompt').value, temp: +$('temp').value, len: 400, id: ++sampleId }));

worker.onmessage = e => {
  const m = e.data;
  if (m.type === 'error') { $('err').textContent = 'Błąd: ' + m.msg; $('err').hidden = false; return; }
  if (m.type === 'sample') { if (m.text != null) $('sample').textContent = $('prompt').value + m.text; return; }
  if (m.type !== 'progress' && m.type !== 'done') return;
  if (m.kind !== cur) return;
  if (m.train != null) series.train.push([m.step, m.train]);
  if (m.val != null && (!series.val.length || series.val[series.val.length - 1][0] !== m.step)) series.val.push([m.step, m.val]);
  live = m;
  drawChart(); renderMetrics(m);
  if (m.sample) $('sample').textContent = m.sample;
  if (m.heat || m.emb) drawViz(m);
  if (m.type === 'done') {
    training = false; $('train').disabled = false; $('stop').disabled = true; $('gen').disabled = false;
    runs.push({ kind: m.kind, label: STEPS.find(s => s.kind === m.kind).name + (m.kind === 'mlp' ? ` (okno ${hpValues().ctx}, emb ${hpValues().emb})` : ''), params: m.params, val: m.val, sample: m.sample, stopped: m.stopped });
    renderCmp();
  }
};

function renderMetrics(m) {
  const V = vocab ? vocab.chars.length : 0, val = m ? m.val : null;
  const rows = [
    ['Błąd sprawdzający', fmt(val, 3), m && m.train != null ? `treningowy ${fmt(m.train, 3)}` : 'mniej = lepiej'],
    ['Bity na znak', fmt(val != null ? val / LN2 : null), `bez wiedzy: ${fmt(Math.log(V) / LN2)}`],
    ['Waha się między', val != null ? `${fmt(Math.exp(val), 1)} literami` : '–', `z ${V} możliwych`],
    ['Krok', m ? m.step.toLocaleString('pl') : '–', m ? `${m.params.toLocaleString('pl')} parametrów` : ''],
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
  const V = vocab ? vocab.chars.length : 2, pts = [...series.train, ...series.val];
  const top = Math.max(Math.log(V), ...pts.map(p => p[1])) * 1.05, maxX = Math.max(1, ...pts.map(p => p[0]));
  const lo = Math.max(0, Math.min(top * .3, ...pts.map(p => p[1])) - .2);
  const x = s => L + (s / maxX) * (W - L - R), y = v => T + (1 - (v - lo) / (top - lo)) * (H - T - B);
  c.font = '11px "Atkinson Hyperlegible",system-ui,sans-serif'; c.fillStyle = css('--muted'); c.strokeStyle = css('--line'); c.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const v = lo + (top - lo) * k / 4, yy = y(v); c.beginPath(); c.moveTo(L, yy); c.lineTo(W - R, yy); c.stroke(); c.textAlign = 'right'; c.textBaseline = 'middle'; c.fillText(fmt(v, 1), L - 6, yy); }
  c.textAlign = 'center'; c.textBaseline = 'top';
  [0, .5, 1].forEach(f => c.fillText(Math.round(maxX * f).toLocaleString('pl'), x(maxX * f), H - B + 8));
  // poziom „bez wiedzy”: każda litera równie prawdopodobna
  const yb = y(Math.log(V)); c.setLineDash([4, 4]); c.strokeStyle = css('--muted'); c.beginPath(); c.moveTo(L, yb); c.lineTo(W - R, yb); c.stroke(); c.setLineDash([]);
  c.textAlign = 'left'; c.textBaseline = 'bottom'; c.fillText('bez wiedzy (losowe litery)', L + 6, yb - 3);
  const line = (arr, col, w) => { if (arr.length < 2) { if (arr.length === 1) { c.fillStyle = col; c.beginPath(); c.arc(x(arr[0][0]), y(arr[0][1]), 3, 0, 7); c.fill(); } return; } c.strokeStyle = col; c.lineWidth = w; c.beginPath(); arr.forEach((p, i) => i ? c.lineTo(x(p[0]), y(p[1])) : c.moveTo(x(p[0]), y(p[1]))); c.stroke(); };
  line(series.train, css('--muted'), 1.4); line(series.val, css('--rot'), 2.2);
  if (!pts.length) { c.fillStyle = css('--muted'); c.textAlign = 'center'; c.textBaseline = 'middle'; c.font = '13px "Atkinson Hyperlegible",system-ui,sans-serif'; c.fillText('Tu pojawi się krzywa błędu.', W / 2, H / 2); }
}

// ---------- wizualizacje ----------
function drawViz(m) {
  const W = 640, H = 420, c = prepCanvas($('viz'), W, H);
  c.clearRect(0, 0, W, H);
  c.font = '13px "Atkinson Hyperlegible",system-ui,sans-serif';
  if (!m || (!m.heat && !m.emb)) {
    $('vizTitle').textContent = cur === 'bigram' ? 'Tabela par liter' : 'Mapa liter';
    $('vizNote').textContent = cur === 'bigram' ? 'Po policzeniu zobaczysz, po jakich literach co zwykle następuje.' : 'Po kilkuset krokach zobaczysz, jak sieć rozmieszcza litery: podobne zachowanie = blisko siebie.';
    c.fillStyle = css('--muted'); c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('Czeka na naukę.', W / 2, H / 2); return;
  }
  if (m.heat) {
    $('vizTitle').textContent = 'Tabela par liter';
    $('vizNote').textContent = 'Wiersz: poprzednia litera. Kolumna: następna. Im jaśniej, tym częściej. Pokazane najczęstsze znaki; ␣ to spacja, ↵ to nowa linia.';
    const { labels, m: M } = m.heat, n = labels.length, pad = 26, cell = Math.min((W - pad - 6) / n, (H - pad - 6) / n);
    c.textAlign = 'center'; c.textBaseline = 'middle'; c.font = '11px "JetBrains Mono",monospace'; c.fillStyle = css('--muted');
    labels.forEach((l, i) => { c.fillText(show(l), pad + cell * i + cell / 2, pad / 2); c.fillText(show(l), pad / 2, pad + cell * i + cell / 2); });
    const rot = [156, 172, 112];
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      const v = Math.sqrt(M[a][b]);       // pierwiastek, żeby rzadkie pary też było widać
      c.fillStyle = `rgba(${rot},${Math.min(1, v * 1.6).toFixed(3)})`;
      c.fillRect(pad + cell * b + .5, pad + cell * a + .5, cell - 1, cell - 1);
    }
    return;
  }
  $('vizTitle').textContent = 'Mapa liter (embeddingi)';
  $('vizNote').textContent = 'Każda litera to punkt; sieć sama ustawia je tak, żeby podobnie zachowujące się litery leżały blisko. Zielone: samogłoski, białe: spółgłoski, niebieskie: odstępy i interpunkcja. Rzut wielu wymiarów na płaszczyznę (PCA), więc odległości są przybliżone.';
  const { labels, pts } = m.emb, xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), P = 26;
  const X = v => P + (v - x0) / ((x1 - x0) || 1) * (W - 2 * P), Y = v => P + (v - y0) / ((y1 - y0) || 1) * (H - 2 * P);
  c.textAlign = 'center'; c.textBaseline = 'middle'; c.font = '15px "JetBrains Mono",monospace';
  labels.forEach((l, i) => {
    const vow = POLISH_VOWELS.has(l), letter = /\p{L}/u.test(l);
    c.fillStyle = vow ? css('--rot') : letter ? css('--fg') : '#7fa6d8';
    c.fillText(show(l), X(pts[i][0]), Y(pts[i][1]));
  });
}

// ---------- porównanie ----------
const runs = [];
function renderCmp() {
  const tb = $('cmp').querySelector('tbody'); tb.innerHTML = '';
  if (!runs.length) { tb.innerHTML = '<tr><td colspan="6" class="note">Tu trafi każda zakończona nauka, żeby porównać konstrukcje na tym samym tekście.</td></tr>'; return; }
  const best = Math.min(...runs.map(r => r.val));
  runs.forEach(r => {
    const tr = document.createElement('tr'); if (r.val === best) tr.className = 'best';
    [r.label + (r.stopped ? ' · przerwana' : ''), r.params.toLocaleString('pl'), fmt(r.val, 3), fmt(r.val / LN2), fmt(Math.exp(r.val), 1), r.sample.slice(0, 60)].forEach((v, i) => {
      const td = document.createElement('td'); td.textContent = v; if (i === 5) td.className = 's'; tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
}

// ---------- start ----------
renderSteps(); renderExplain(); renderHP(); renderCmp();
loadText(SAMPLE_TEXT, 'przykład (Claude)');
