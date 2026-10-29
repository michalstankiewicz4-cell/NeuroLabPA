// Laboratorium języka · wątek treningowy. Liczy w tle, żeby strona nigdy się nie zawieszała.
importScripts('engine.js', 'wasm.js', 'gpu.js', 'bpe.js');

let tmask = null, ftSaved = null;   // maska celów (dostrajanie do rozmowy) i odłożone dane głównego tekstu
const tgt = i => tmask && !tmask[i] ? -1 : tokens[i];
let tokens = null, V = 0, chars = [], nTrain = 0, valIdx = null, nlIdx = 0, dataLabel = '', tok = null, topIdx = [], cpt = 1;
let running = null;                 // aktualny trening: {kind, stop}
const models = {};                  // wytrenowane modele, do próbek, zapisu i porównań

const post = m => self.postMessage(m);
const randInt = n => Math.floor(Math.random() * n);
const now = () => performance.now();

// Jednostka tekstu: litery albo tokeny BPE. `chars` to zawsze słownik jednostek (litery albo kawałki słów).
function setData(text, ch, label, unit) {
  dataLabel = label || '';
  const nm = /^bpe(\d+)$/.exec(unit || '');
  tok = nm ? bpeTrain(text, +nm[1], (r, n) => post({ type: 'tokprogress', r, n })) : charTokenizer(ch);
  chars = tok.vocab; V = chars.length;
  tokens = Int32Array.from(tok.encode(text));
  nTrain = Math.floor(tokens.length * .9);           // ostatnie 10% tekstu zostaje do sprawdzania (walidacja)
  const nv = Math.min(3000, tokens.length - nTrain - 1);
  valIdx = Int32Array.from({ length: nv }, () => nTrain + 1 + randInt(tokens.length - nTrain - 1));
  nlIdx = Math.max(0, chars.indexOf('\n'));
  for (const k in models) delete models[k];
  // ile znaków przypada średnio na jednostkę w tekście sprawdzającym: do przeliczania błędu na literę
  let vc = 0; for (let i = nTrain; i < tokens.length; i++) vc += chars[tokens[i]].length;
  cpt = vc / (tokens.length - nTrain);
  const freq = new Float64Array(V); for (let i = 0; i < nTrain; i++) freq[tokens[i]]++;
  topIdx = [...Array(V).keys()].sort((a, b) => freq[b] - freq[a]).slice(0, 150);
  const info = { V, cpt, n: tokens.length, train: nTrain, unit: nm ? 'bpe' : 'char', merges: [], longest: [], example: [] };
  if (nm) {
    info.merges = tok.merges.slice(0, 60).map(([a, b]) => [chars[a], chars[b]]);
    info.longest = chars.slice(tok.nBase).sort((a, b) => b.length - a.length || a.localeCompare(b, 'pl')).filter(t => /\p{L}/u.test(t)).slice(0, 24);
    const ex = tokens.slice(nTrain + 1, nTrain + 400); let st = 0;
    for (let i = 0; i < ex.length; i++) if (chars[ex[i]] === '\n') { st = i + 1; break; }
    info.example = Array.from(ex.slice(st, st + 90), i => chars[i]);
  }
  return info;
}
// Mapa jednostek: rzut embeddingów, tylko 150 najczęstszych (przy tokenach jest ich ponad tysiąc).
function embOf(m, dim) { const pts = pca2(m.E.d, m.chars.length, dim), idx = topIdx.filter(i => i < m.chars.length); return { labels: idx.map(i => m.chars[i]), pts: idx.map(i => pts[i]) }; }

// Losowanie kolejnej litery z rozkładu, z "temperaturą": niższa = bezpieczniej, wyższa = odważniej.
function pick(probs, temp) {
  const w = Array.from(probs, p => Math.pow(Math.max(p, 1e-12), 1 / temp)), s = w.reduce((a, b) => a + b, 0);
  let r = Math.random() * s;
  for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return i; }
  return w.length - 1;
}
function softmaxRow(lg) { let mx = -Infinity; for (const v of lg) mx = Math.max(mx, v); const e = Array.from(lg, v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }

// Każdy model ma własny słownik (chars), bo wczytany z pliku mógł powstać na innym tekście.
// stepper() zwraca funkcję: podaj cały dotychczasowy tekst (numery liter), dostaniesz rozkład następnej litery.
// Przy tokenach końcowa spacja początku tekstu zostaje odcięta: w nauce spacja należy do następnego słowa.
function sampleText(m, prompt, temp, len, stop) {
  const tk = m.tok || charTokenizer(m.chars), bpe = tk.merges.length > 0;
  const ctx = tk.encode(bpe ? prompt.replace(/[^\S\n]+$/, '') : prompt), out = [], next = m.stepper();
  let n = 0;
  for (let i = 0; i < len && n < len; i++) {
    const t = pick(next(ctx), temp), piece = m.chars[t]; ctx.push(t); out.push(piece); n += piece.length;
    if (stop && stop(out.join(''))) break;
  }
  return out.join('');
}

// ---------- krok 1: bigram (liczenie par liter) ----------
function bigramModel(P, ch) {
  const Vm = ch.length, nl = Math.max(0, ch.indexOf('\n'));
  return { kind: 'bigram', hp: {}, chars: ch, P, count: Vm * Vm,
    stepper: () => ctx => P.subarray((ctx.length ? ctx[ctx.length - 1] : nl) * Vm, (ctx.length ? ctx[ctx.length - 1] : nl) * Vm + Vm),
    tensors: () => [{ d: P }] };
}
function trainBigram() {
  const C = new Float64Array(V * V), alpha = .5;      // wygładzanie: para, której nie było, nie dostaje zera
  for (let i = 1; i < nTrain; i++) C[tokens[i - 1] * V + tokens[i]]++;
  const P = new Float32Array(V * V);
  for (let a = 0; a < V; a++) {
    let s = 0; for (let b = 0; b < V; b++) s += C[a * V + b] + alpha;
    for (let b = 0; b < V; b++) P[a * V + b] = (C[a * V + b] + alpha) / s;
  }
  const lossOn = (from, to) => { let l = 0, n = 0; for (let i = Math.max(1, from); i < to; i++) { l -= Math.log(P[tokens[i - 1] * V + tokens[i]]); n++; } return l / n; };
  const model = bigramModel(P, chars); model.tok = tok;
  model.val = lossOn(nTrain + 1, tokens.length); model.label = dataLabel;
  models.bigram = model;
  const freq = new Float64Array(V); for (let i = 0; i < nTrain; i++) freq[tokens[i]]++;
  const top = [...Array(V).keys()].sort((a, b) => freq[b] - freq[a]).slice(0, Math.min(28, V));
  const res = { kind: 'bigram', step: 1, train: lossOn(1, nTrain), val: model.val, params: model.count, bestStep: 1,
    sample: sampleText(model, '', .9, 240), heat: { labels: top.map(i => chars[i]), m: top.map(a => top.map(b => P[a * V + b])) } };
  post({ type: 'progress', ...res }); post({ type: 'done', ...res });
}

// ---------- krok 2: sieć z oknem (Bengio i in., 2003) ----------
function buildMLP(hp, ch) {
  const { ctx, emb, hidden } = hp, Vm = ch.length, nl = Math.max(0, ch.indexOf('\n'));
  const E = param([Vm, emb], 1);
  const W1 = param([ctx * emb, hidden], (5 / 3) / Math.sqrt(ctx * emb));   // skala zgodna z tanh, żeby neurony nie nasycały się od startu
  const b1 = zeros([hidden]);
  const W2 = param([hidden, Vm], .01);                                      // małe wyjście = na starcie wszystkie litery równie prawdopodobne
  const b2 = zeros([Vm]);
  const params = [E, W1, b1, W2, b2];
  const forward = idx => {             // idx: Int32Array długości B*ctx
    const B = idx.length / ctx;
    const x = view(gather(E, idx), [B, ctx * emb]);
    const h = tanh(addBias(matmul(x, W1), b1));
    return addBias(matmul(h, W2), b2);
  };
  return { kind: 'mlp', hp, chars: ch, E, params, forward, count: params.reduce((s, p) => s + p.size, 0),
    stepper: () => c => {
      const idx = new Int32Array(ctx);
      for (let j = 0; j < ctx; j++) { const k = c.length - ctx + j; idx[j] = k >= 0 ? c[k] : nl; }
      return softmaxRow(forward(idx).d);
    } };
}
function mlpBatch(positions, ctx) {
  const X = new Int32Array(positions.length * ctx), Y = new Int32Array(positions.length);
  positions.forEach((p, b) => { for (let j = 0; j < ctx; j++) { const k = p - ctx + j; X[b * ctx + j] = k >= 0 ? tokens[k] : nlIdx; } Y[b] = tgt(p); });
  return { X, Y };
}
function evalMLP(m, idxs) {
  let loss = 0, n = 0; const ctx = m.hp.ctx, chunk = 500;
  for (let i = 0; i < idxs.length; i += chunk) {
    const part = Array.from(idxs.slice(i, i + chunk)), { X, Y } = mlpBatch(part, ctx), ce = crossEntropy(m.forward(X), Y);
    loss += ce.d[0] * ce.nValid; n += ce.nValid;
  }
  return n ? loss / n : 0;
}

// ---------- krok 3: sieć rekurencyjna (zwykła RNN albo GRU) ----------
// Czyta tekst litera po literze i niesie pamięć h. Wszystkie litery przechodzą przez te same wagi,
// więc długość kontekstu nie dokłada parametrów (w przeciwieństwie do okna z kroku 2).
function buildRNN(hp, ch) {
  const { cell, emb, hidden: H } = hp, Vm = ch.length, k = cell === 'gru' ? 3 : 1, nl = Math.max(0, ch.indexOf('\n'));
  const E = param([Vm, emb], 1);
  const W = param([emb, k * H], 1 / Math.sqrt(emb));
  const b = zeros([k * H]);
  const U = param([H, k * H], 1 / Math.sqrt(H));
  const bhn = zeros([H]);
  const Wo = param([H, Vm], .01), bo = zeros([Vm]);
  const params = cell === 'gru' ? [E, W, b, U, bhn, Wo, bo] : [E, W, b, U, Wo, bo];
  const step = (xw, h) => cell === 'gru' ? gruCell(xw, h, U, bhn) : rnnCell(xw, h, U);
  // X: numery liter ułożone krokami czasu: X[t*B + i] to litera t w ciągu i. Zwraca pamięć po każdej literze.
  const run = (X, B, h0) => {
    const Tn = X.length / B, xw = addBias(matmul(gather(E, X), W), b), hs = [];
    let h = h0 || new T(new Float32Array(B * H), [B, H]);
    for (let t = 0; t < Tn; t++) { h = step(sliceRows(xw, t * B, B), h); hs.push(h); }
    return hs;
  };
  const out = hs => addBias(matmul(hs.length === 1 ? hs[0] : concatRows(hs), Wo), bo);
  return { kind: 'rnn', hp, chars: ch, E, params, run, out, count: params.reduce((s, p) => s + p.size, 0), H,
    stepper: () => {
      let h = run(Int32Array.of(nl), 1)[0], seen = 0;      // start jak po nowej linii
      return c => { while (seen < c.length) { h = run(Int32Array.of(c[seen]), 1, h)[0]; seen++; } return softmaxRow(out([h]).d); };
    } };
}
// Strumieniowe sprawdzanie: kilka kawałków tekstu czytanych od początku; pierwsze `warm` liter tylko
// rozgrzewa pamięć i nie liczy się do błędu. Tak mierzymy to, co sieć potrafi przy czytaniu ciągłym.
function evalRNN(m, from, to, S = 24, warm = 40, count = 160) {
  const L = warm + count, span = to - from - L - 1, X = new Int32Array(L * S), Y = new Int32Array(count * S);
  for (let i = 0; i < S; i++) {
    const s = from + Math.floor(span * (i + .5) / S);
    for (let t = 0; t < L; t++) X[t * S + i] = tokens[s + t];
    for (let t = 0; t < count; t++) Y[t * S + i] = tgt(s + warm + t + 1);
  }
  const hs = m.run(X, S);
  return crossEntropy(m.out(hs.slice(warm)), Y).d[0];
}
// Jak daleko sięga pamięć: błąd liczymy tylko na ostatniej literze i patrzymy, jak silny sygnał (gradient)
// dociera do pamięci sprzed 1, 2, … liter. W zwykłej RNN maleje wykładniczo („zanikający gradient”).
function gradReach(m, L = 60, S = 8) {
  const X = new Int32Array(L * S), Y = new Int32Array(S);
  for (let i = 0; i < S; i++) { const s = 1 + randInt(nTrain - L - 2); for (let t = 0; t < L; t++) X[t * S + i] = tokens[s + t]; Y[i] = tokens[s + L]; }
  const hs = m.run(X, S), loss = crossEntropy(m.out([hs[L - 1]]), Y);
  const keep = m.params.map(p => p.g); m.params.forEach(p => { p.g = new Float32Array(p.size); });
  backward(loss);
  const norms = hs.map(h => Math.hypot(...h.g));
  m.params.forEach((p, i) => { p.g = keep[i]; });
  const base = norms[L - 1] || 1;
  return norms.reverse().map(v => v / base);         // [0] = ostatnia litera, [d] = d liter wcześniej
}
// Co robią poszczególne neurony pamięci: aktywacje na kawałku tekstu i neurony, które najmocniej
// pokrywają się z prostymi cechami tekstu (sieci nikt tych cech nie podawał).
function neuronProbe(m, show = 320, scan = 2500) {
  const from = nTrain + 1, start = from + randInt(Math.max(1, tokens.length - from - scan - 2));
  const len = Math.min(scan, tokens.length - start - 1), X = tokens.subarray(start, start + len);
  const H = m.H, acts = new Float32Array(len * H);
  let h = null;
  for (let t = 0; t < len; t++) { h = m.run(Int32Array.of(X[t]), 1, h)[0]; acts.set(h.d, t * H); }
  const isL = c => /\p{L}/u.test(c), feats = { 'litera (nie odstęp)': [], 'pozycja w słowie': [], 'w cudzysłowie': [], 'w nawiasie': [], 'samogłoska': [] };
  let pos = 0, q = 0, par = 0;
  for (let t = 0; t < len; t++) {
    const piece = m.chars[X[t]], last = piece[piece.length - 1];   // przy tokenach: stan po całym kawałku
    pos = /^[^\p{L}]/u.test(piece) ? (isL(last) ? 1 : 0) : (isL(last) ? pos + 1 : 0);
    for (const c of piece) {
      if (c === '„' || c === '«') q = 1; else if (c === '”' || c === '»') q = 0; else if (c === '"') q = 1 - q;
      if (c === '(') par++; else if (c === ')') par = Math.max(0, par - 1);
      if (c === '\n') { q = 0; par = 0; }
    }
    feats['litera (nie odstęp)'].push(isL(last) ? 1 : 0); feats['pozycja w słowie'].push(Math.min(pos, 10));
    feats['w cudzysłowie'].push(q); feats['w nawiasie'].push(par > 0 ? 1 : 0); feats['samogłoska'].push('aąeęioóuyAĄEĘIOÓUY'.includes(last) ? 1 : 0);
  }
  const found = [];
  for (const [name, f] of Object.entries(feats)) {
    const n = f.length, mf = f.reduce((a, b) => a + b, 0) / n, vf = f.reduce((a, b) => a + (b - mf) ** 2, 0);
    if (vf < n * .01) continue;                       // cecha prawie nie występuje w tym kawałku
    let best = null;
    for (let j = 0; j < H; j++) {
      let ma = 0; for (let t = 0; t < n; t++) ma += acts[t * H + j]; ma /= n;
      let cov = 0, va = 0; for (let t = 0; t < n; t++) { const a = acts[t * H + j] - ma; cov += a * (f[t] - mf); va += a * a; }
      const r = cov / Math.sqrt(va * vf + 1e-12);
      if (!best || Math.abs(r) > Math.abs(best.r)) best = { name, neuron: j, r };
    }
    found.push(best);
  }
  const s0 = Math.max(0, len - show);                  // pokazujemy końcówkę: pamięć jest już „rozgrzana”
  return { pieces: Array.from(X.subarray(s0), i => m.chars[i]), H, acts: acts.slice(s0 * H), found };
}
function rnnBatch(B, Tn) {
  const X = new Int32Array(Tn * B), Y = new Int32Array(Tn * B);
  for (let i = 0; i < B; i++) { const s = randInt(nTrain - Tn - 1); for (let t = 0; t < Tn; t++) { X[t * B + i] = tokens[s + t]; Y[t * B + i] = tgt(s + t + 1); } }
  return { X, Y };
}


// ---------- krok 4: transformer (Vaswani i in. 2017; tu wersja „tylko dekoder”, jak w GPT) ----------
// Każda pozycja w oknie T liter jednocześnie patrzy na wszystkie wcześniejsze (uwaga) i sama wybiera,
// które są ważne. Nic nie musi przejść przez wąską pamięć jak w RNN, a wszystkie pozycje liczą się naraz.
function buildTransformer(hp, ch) {
  const { layers: L, dim: d, heads: nh, ctx: T } = hp, Vm = ch.length, nl = Math.max(0, ch.indexOf('\n'));
  const one = n => { const t = zeros([n]); t.d.fill(1); return t; };
  const Emb = param([Vm, d], .3), P = param([T, d], .1), params = [Emb, P], blocks = [];
  const s = 1 / Math.sqrt(d), sr = s / Math.sqrt(2 * L);     // wyjścia bloków mniejsze: suma wielu warstw nie rośnie
  for (let l = 0; l < L; l++) {
    const b = { g1: one(d), b1: zeros([d]), Wqkv: param([d, 3 * d], s), bqkv: zeros([3 * d]), Wp: param([d, d], sr), bp: zeros([d]),
      g2: one(d), b2: zeros([d]), W1: param([d, 4 * d], s), c1: zeros([4 * d]), W2: param([4 * d, d], sr / 2), c2: zeros([d]) };
    blocks.push(b); params.push(...Object.values(b));
  }
  const gf = one(d), bf = zeros([d]), Wo = param([d, Vm], .01), bo = zeros([Vm]);
  params.push(gf, bf, Wo, bo);
  // X: B okien po Tw liter, wiersz b*Tw + t. Zwraca logity dla każdej pozycji i tensory uwagi z każdej warstwy.
  const forward = (X, B, Tw) => {
    let x = addPos(gather(Emb, X), P, Tw); const atts = [];
    for (const b of blocks) {
      const at = causalAttention(addBias(matmul(layerNorm(x, b.g1, b.b1), b.Wqkv), b.bqkv), B, Tw, nh); atts.push(at);
      x = add(x, addBias(matmul(at, b.Wp), b.bp));                                   // uwaga + połączenie omijające (residual)
      x = add(x, addBias(matmul(gelu(addBias(matmul(layerNorm(x, b.g2, b.b2), b.W1), b.c1)), b.W2), b.c2));   // sieć w każdej pozycji osobno
    }
    return { logits: addBias(matmul(layerNorm(x, gf, bf), Wo), bo), atts };
  };
  return { kind: 'transformer', hp, chars: ch, E: Emb, params, forward, count: params.reduce((a, p) => a + p.size, 0),
    stepper: () => c => {
      const w = c.length ? c.slice(-T) : [nl], X = Int32Array.from(w);
      const lg = forward(X, 1, X.length).logits;
      return softmaxRow(lg.d.subarray((X.length - 1) * Vm, X.length * Vm));
    } };
}
function tfBatch(B, T, from, to) {
  const X = new Int32Array(B * T), Y = new Int32Array(B * T);
  for (let b = 0; b < B; b++) { const s = from + randInt(to - from - T - 1); for (let t = 0; t < T; t++) { X[b * T + t] = tokens[s + t]; Y[b * T + t] = tgt(s + t + 1); } }
  return { X, Y };
}
// Sprawdzanie: okna z tekstu sprawdzającego; liczą się tylko pozycje z drugiej połowy okna,
// żeby każda przewidywana litera miała przed sobą co najmniej T/2 liter (jak rozgrzanie w RNN).
function evalTF(m, from, to, S = 32) {
  const T = m.hp.ctx, Vm = m.chars.length, span = to - from - T - 2, X = new Int32Array(S * T), Y = new Int32Array(S * T);
  for (let b = 0; b < S; b++) { const s = from + Math.floor(span * (b + .5) / S); for (let t = 0; t < T; t++) { X[b * T + t] = tokens[s + t]; Y[b * T + t] = tgt(s + t + 1); } }
  const lg = m.forward(X, S, T).logits.d; let loss = 0, n = 0;
  for (let b = 0; b < S; b++) for (let t = T >> 1; t < T; t++) {
    if (Y[b * T + t] < 0) continue;
    const o = (b * T + t) * Vm; let mx = -Infinity; for (let j = 0; j < Vm; j++) mx = Math.max(mx, lg[o + j]);
    let sum = 0; for (let j = 0; j < Vm; j++) sum += Math.exp(lg[o + j] - mx);
    loss += Math.log(sum) + mx - lg[o + Y[b * T + t]]; n++;
  }
  return n ? loss / n : 0;
}
// Mapa uwagi: jedno okno tekstu sprawdzającego, wagi uwagi z każdej warstwy i głowy
// oraz to, jak daleko wstecz każda głowa średnio patrzy.
function attentionProbe(m) {
  const T = m.hp.ctx, nh = m.hp.heads, s = nTrain + 1 + randInt(tokens.length - nTrain - T - 2), X = tokens.slice(s, s + T);
  const { atts } = m.forward(X, 1, T);
  const heads = [];
  atts.forEach((a, l) => { for (let h = 0; h < nh; h++) {
    let dist = 0; for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) dist += a.att[h * T * T + i * T + j] * (i - j);
    heads.push({ layer: l, head: h, dist: dist / T });
  } });
  return { pieces: Array.from(X, i => m.chars[i]), T, L: atts.length, nh, att: atts.map(a => a.att), heads };
}
function trainTF(hp) { const m = buildTransformer(hp, chars); trainLoop(m, hp, fnsTF(m, hp)); }
function fnsTF(m, hp) {
  return ({
    loss: () => { const { X, Y } = tfBatch(hp.batch, hp.ctx, 0, nTrain); return crossEntropy(m.forward(X, hp.batch, hp.ctx).logits, Y); },
    val: () => evalTF(m, nTrain + 1, tokens.length), train: () => evalTF(m, 1, nTrain), valMs: 2500, clip: 1,
    extras: () => ({ attn: attentionProbe(m), emb: embOf(m, hp.dim) }),
  });
}

// ---------- transformer na karcie graficznej ----------
let gpuEng = null;
async function getGPU() { if (!gpuEng || gpuEng.lost) gpuEng = await GPUEngine.create(); return gpuEng; }
function tfEvalXY(T, from, to, S = 32) {
  const span = to - from - T - 2, X = new Int32Array(S * T), Y = new Int32Array(S * T);
  for (let b = 0; b < S; b++) { const s = from + Math.floor(span * (b + .5) / S); for (let t = 0; t < T; t++) { X[b * T + t] = tokens[s + t]; Y[b * T + t] = tokens[s + t + 1]; } }
  return { X, Y };
}
async function trainTFGPU(hp) {
  const eng = await getGPU(), m = buildTransformer(hp, chars), job = { kind: 'transformer', stop: false };
  running = job; models.transformer = m; m.label = dataLabel;
  const T = hp.ctx, S = 32, g = new GPUTransformer(eng, m, hp.batch, S);
  const vXY = tfEvalXY(T, nTrain + 1, tokens.length, S), tXY = tfEvalXY(T, 1, nTrain, S);
  const evalLoss = async ({ X, Y }) => { const r = await g.evalRows(X, Y); let l = 0, n = 0; for (let b = 0; b < S; b++) for (let t = T >> 1; t < T; t++) { l += r[b * T + t]; n++; } return l / n; };
  const extras = () => ({ attn: attentionProbe(m), emb: embOf(m, hp.dim) });
  let step = 0, ema = null, lastVal = await evalLoss(vXY), best = { val: lastVal, step: 0 }, tVal = now(), tExtra = now(), k = 1, ms = null;
  g.snapshot();
  const base = () => ({ kind: 'transformer', step, train: ema, val: lastVal, params: m.count, bestStep: best.step, bestVal: best.val, ms, device: 'GPU · ' + eng.name });
  post({ type: 'progress', ...base(), sample: sampleText(m, '', .8, 240), ...extras() });
  while (!job.stop && step < hp.steps) {
    const n = Math.min(k, hp.steps - step), t0 = now();
    const lr = hp.lr * (step > hp.steps * .7 ? .2 : 1);
    g.steps(Array.from({ length: n }, () => tfBatch(hp.batch, T, 0, nTrain)), lr, 1);
    const L = await g.lastLoss(); step += n;            // odczyt błędu czeka, aż karta skończy
    ms = (now() - t0) / n; k = Math.max(1, Math.min(64, Math.round(150 / ms)));
    ema = ema == null ? L : ema * Math.pow(.97, n) + L * (1 - Math.pow(.97, n));
    const msg = { type: 'progress' };
    if (now() - tVal > 1500) { tVal = now(); lastVal = await evalLoss(vXY); if (lastVal < best.val - 1e-4) { best = { val: lastVal, step }; g.snapshot(); } }
    if (now() - tExtra > 9000) { tExtra = now(); await g.pull(); Object.assign(msg, { sample: sampleText(m, '', .8, 240) }, extras()); }
    post({ ...msg, ...base() });
    await new Promise(r => setTimeout(r, 0));           // miejsce na wiadomość „stop”
  }
  let val = await evalLoss(vXY), restored = false;
  if (val < best.val - 1e-4) best = { val, step };
  else if (val > best.val + .01) { g.restore(); val = best.val; restored = true; }
  await g.pull();
  m.val = val; m.step = step; m.bestStep = best.step;
  post({ type: 'done', ...base(), val, train: await evalLoss(tXY), restored, sample: sampleText(m, '', .8, 300), stopped: job.stop, hp, ...extras() });
  running = null;
}
// Zgodność: ten sam start i ta sama porcja tekstu na CPU i GPU, jeden krok nauki, porównanie wyników.
async function gpuSelfTest() {
  const eng = await getGPU(), hp = { layers: 2, dim: 32, heads: 4, ctx: 32, lr: .003 }, B = 4;
  const mC = buildTransformer(hp, chars), mG = buildTransformer(hp, chars);
  mG.params.forEach((p, i) => p.d.set(mC.params[i].d));
  const { X, Y } = tfBatch(B, hp.ctx, 0, nTrain);
  const opt = new Adam(mC.params, hp.lr), loss = crossEntropy(mC.forward(X, B, hp.ctx).logits, Y);
  opt.zeroGrad(); backward(loss); opt.clip(1); opt.step();
  eng.dev.pushErrorScope('validation');
  const g = new GPUTransformer(eng, mG, B, 1);
  g.steps([{ X, Y }], hp.lr, 1);
  const err = await eng.dev.popErrorScope();
  if (err) throw new Error('WebGPU: ' + err.message);
  const lossG = await g.lastLoss(); await g.pull();
  let diff = 0; mG.params.forEach((p, i) => { for (let j = 0; j < p.size; j++) diff = Math.max(diff, Math.abs(p.d[j] - mC.params[i].d[j])); });
  return { name: eng.name, lossC: loss.d[0], lossG, diff };
}

// ---------- krok 5: dostrajanie do rozmowy ----------
// Bierzemy kopię wytrenowanej sieci i uczymy ją dalej na dialogach „Ty: … / Sieć: …”. Przy maskowaniu
// błąd liczy się tylko na kwestiach Sieci: sieć ma się nauczyć odpowiadać, a nie mówić za rozmówcę.
function restoreData() { if (ftSaved) { ({ tokens, nTrain, valIdx, chars, V, tok, cpt, topIdx, nlIdx } = ftSaved); ftSaved = null; } tmask = null; }
function finetune(baseKind, fhp, dialog) {
  restoreData();
  const b = models[baseKind];
  if (!b || !b.params) throw new Error('Najpierw naucz albo wczytaj sieć, którą chcesz dostroić (krok 2, 3 albo 4).');
  const m = b.kind === 'mlp' ? buildMLP(b.hp, b.chars) : b.kind === 'rnn' ? buildRNN(b.hp, b.chars) : buildTransformer(b.hp, b.chars);
  m.params.forEach((p, i) => p.d.set(b.params[i].d)); m.slot = 'chat'; m.base = baseKind;
  ftSaved = { tokens, nTrain, valIdx, chars, V, tok, cpt, topIdx, nlIdx };
  tok = b.tok || charTokenizer(b.chars); chars = b.chars; V = chars.length; nlIdx = Math.max(0, chars.indexOf('\n'));
  const known = new Set(tok.vocab.slice(0, tok.nBase));
  const convs = [...dialog.replace(/\r\n?/g, '\n')].filter(c => known.has(c)).join('').split(/\n\s*\n/).map(c => c.trim()).filter(Boolean);
  const nv = Math.max(4, Math.round(convs.length * .1));
  const trainText = convs.slice(0, -nv).join('\n\n') + '\n\n', full = trainText + convs.slice(-nv).join('\n\n') + '\n';
  // maska znaków: 1 dla treści kwestii Sieci (po „Sieć:”, razem z końcem linii), 0 dla reszty
  const cm = new Uint8Array(full.length); let ls = 0;
  for (let i = 0; i <= full.length; i++) if (i === full.length || full[i] === '\n') {
    if (full.startsWith('Sieć:', ls)) for (let j = ls + 5; j <= Math.min(i, full.length - 1); j++) cm[j] = 1;
    ls = i + 1;
  }
  const ids = tok.encode(full); tokens = Int32Array.from(ids); nTrain = tok.encode(trainText).length;
  const tm = new Uint8Array(ids.length); let pos = 0;
  ids.forEach((t, i) => { tm[i] = cm[pos] || 0; pos += chars[t].length; });
  tmask = fhp.mask ? tm : null;
  const vpos = []; for (let i = nTrain + 1; i < tokens.length; i++) if (!tmask || tmask[i]) vpos.push(i);
  valIdx = Int32Array.from(vpos.slice(0, 3000));
  let vc = 0; for (let i = nTrain; i < tokens.length; i++) vc += chars[tokens[i]].length; cpt = vc / (tokens.length - nTrain);
  const freq = new Float64Array(V); for (let i = 0; i < nTrain; i++) freq[tokens[i]]++;
  topIdx = [...Array(V).keys()].sort((a, c) => freq[c] - freq[a]).slice(0, 150);
  post({ type: 'ftinfo', cpt, conv: convs.length, val: nv, tokens: tokens.length, learn: tm.reduce((a, c) => a + c, 0) });
  const hp = { ...b.hp, steps: fhp.steps, lr: fhp.lr, device: fhp.device, base: baseKind, mask: fhp.mask };
  m.label = `rozmowy (${convs.length - nv} do nauki, ${nv} do sprawdzania)`;
  const fns = b.kind === 'mlp' ? fnsMLP(m, hp) : b.kind === 'rnn' ? fnsRNN(m, hp) : fnsTF(m, hp);
  fns.prompt = 'Ty: Cześć! Jak się masz?\nSieć:'; fns.done = restoreData;
  trainLoop(m, hp, fns);
}
// Odpowiedź w czacie: historia rozmowy + „Ty: …\nSieć:”, losujemy do końca linii.
function chatReply(slot, history, temp) {
  const m = models[slot];
  if (!m) throw new Error('Brak sieci do rozmowy.');
  // początkowe odstępy i nowe linie pomijamy; kwestia kończy się na pierwszym końcu linii po treści
  const raw = sampleText(m, history, temp, 260, out => out.trimStart().includes('\n'));
  return raw.trimStart().split('\n')[0].trim();
}

// ---------- wspólna pętla nauki (krok 2, 3 i 4) ----------
// Pamięta najlepszą wersję sieci: gdy błąd sprawdzający zaczyna rosnąć (uczenie na pamięć),
// na koniec wracamy do wag z najlepszego momentu.
// Sposób liczenia na procesorze: 'wasm' (WebAssembly SIMD) albo zwykły JS.
function cpuBackend(hp) {
  try { return setBackend(hp.device === 'cpu' ? 'js' : 'wasm') === 'wasm' ? 'WASM SIMD' : 'JS'; }
  catch (err) { post({ type: 'error', msg: err.message + ' Liczę zwykłym JS.' }); setBackend('js'); return 'JS'; }
}
function trainLoop(m, hp, fns) {
  const devName = cpuBackend(hp), slot = m.slot || m.kind; m.tok = tok;
  const opt = new Adam(m.params, hp.lr), job = { kind: slot, stop: false };
  running = job; models[slot] = m; m.label = m.label || dataLabel;
  let step = 0, ema = null, lastVal = null, best = null, tVal = 0, tExtra = 0, ms = null;
  const snap = () => m.params.map(p => p.d.slice());
  const report = (extras, withVal) => {
    if (withVal) {
      lastVal = fns.val();
      if (!best || lastVal < best.val - 1e-4) best = { val: lastVal, step, w: snap() };
    }
    const msg = { type: 'progress', kind: slot, step, train: ema, val: lastVal, params: m.count, bestStep: best && best.step, bestVal: best && best.val, ms, device: devName };
    if (extras) Object.assign(msg, { sample: sampleText(m, fns.prompt || '', .8, 240) }, fns.extras());
    post(msg);
  };
  report(true, true);
  const tick = () => {
    if (job.stop || step >= hp.steps) { finish(); return; }
    const t0 = now(), s0 = step;
    while (now() - t0 < 60 && step < hp.steps) {
      const loss = fns.loss();
      opt.zeroGrad(); backward(loss);
      if (fns.clip) opt.clip(fns.clip);
      opt.lr = hp.lr * (step > hp.steps * .7 ? .2 : 1);   // pod koniec mniejsze kroki, żeby dopracować szczegóły
      opt.step(); step++;
      ema = ema == null ? loss.d[0] : ema * .97 + loss.d[0] * .03;
    }
    ms = (now() - t0) / (step - s0);
    const t = now(), doVal = t - tVal > fns.valMs, doExtra = t - tExtra > 9000;
    if (doVal) tVal = t; if (doExtra) tExtra = t;
    report(doExtra, doVal);
    setTimeout(tick, 0);
  };
  const finish = () => {
    let val = fns.val(), restored = false;
    if (!best || val < best.val - 1e-4) best = { val, step, w: snap() };
    else if (val > best.val + .01) {   // mniejsze różnice to szum pomiaru, nie uczenie na pamięć
    m.params.forEach((p, i) => p.d.set(best.w[i])); val = best.val; restored = true; }
    m.val = val; m.step = step; m.bestStep = best.step;
    post({ type: 'done', kind: slot, step, train: fns.train(), val, params: m.count, bestStep: best.step, bestVal: best.val, restored,
      sample: sampleText(m, fns.prompt || '', .8, 300), stopped: job.stop, hp, ms, device: devName, ...fns.extras() });
    running = null;
    if (fns.done) fns.done();
  };
  setTimeout(tick, 0);
}

function trainMLP(hp) { const m = buildMLP(hp, chars); trainLoop(m, hp, fnsMLP(m, hp)); }
function fnsMLP(m, hp) {
  const trainCheck = Int32Array.from({ length: Math.min(3000, nTrain - 1) }, () => 1 + randInt(nTrain - 1));
  return ({
    loss: () => { const pos = Array.from({ length: hp.batch }, () => 1 + randInt(nTrain - 1)), { X, Y } = mlpBatch(pos, hp.ctx); return crossEntropy(m.forward(X), Y); },
    val: () => evalMLP(m, valIdx), train: () => evalMLP(m, trainCheck), valMs: 1500,
    extras: () => ({ emb: embOf(m, hp.emb) }),
  });
}
function trainRNN(hp) { const m = buildRNN(hp, chars); trainLoop(m, hp, fnsRNN(m, hp)); }
function fnsRNN(m, hp) {
  return ({
    loss: () => { const { X, Y } = rnnBatch(hp.batch, hp.seq); return crossEntropy(m.out(m.run(X, hp.batch)), Y); },
    val: () => evalRNN(m, nTrain + 1, tokens.length), train: () => evalRNN(m, 1, nTrain), valMs: 2500, clip: 1,
    extras: () => ({ reach: gradReach(m), neurons: neuronProbe(m), emb: embOf(m, hp.emb) }),
  });
}

// ---------- zapis i wczytanie modelu ----------
const b64 = f => { const u = new Uint8Array(f.buffer, f.byteOffset, f.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
const unb64 = s => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); };
function exportModel(kind) {
  const m = models[kind];
  if (!m) throw new Error('Brak wytrenowanego modelu tego rodzaju.');
  const tensors = m.kind === 'bigram' ? [m.P] : m.params.map(p => p.d);
  return { format: 'neurolab-lang-model', version: 1, kind: m.kind, hp: m.hp, chars: m.chars, tok: m.tok && m.tok.merges.length ? { merges: m.tok.merges, nBase: m.tok.nBase } : null, trainedOn: m.label || '',
    val: m.val ?? null, step: m.step ?? null, params: m.count, savedAt: new Date().toISOString(), tensors: tensors.map(b64) };
}
function importModel(j) {
  if (!j || j.format !== 'neurolab-lang-model') throw new Error('To nie jest plik modelu z Laboratorium języka.');
  const t = j.tensors.map(unb64); let m;
  if (j.kind === 'bigram') m = bigramModel(t[0], j.chars);
  else {
    m = j.kind === 'mlp' ? buildMLP(j.hp, j.chars) : j.kind === 'transformer' ? buildTransformer(j.hp, j.chars) : buildRNN(j.hp, j.chars);
    if (t.length !== m.params.length || t.some((a, i) => a.length !== m.params[i].size)) throw new Error('Plik modelu jest niekompletny albo z innej wersji.');
    m.params.forEach((p, i) => p.d.set(t[i]));
  }
  m.label = j.trainedOn; m.step = j.step;
  m.tok = j.tok ? bpeTokenizer(j.chars, j.tok.merges, j.tok.nBase) : charTokenizer(j.chars);
  const same = j.chars.length === chars.length && j.chars.every((c, i) => c === chars[i]);
  let val = null;
  if (same) {
    if (m.kind === 'bigram') { let l = 0, n = 0; for (let i = nTrain + 1; i < tokens.length; i++) { l -= Math.log(m.P[tokens[i - 1] * V + tokens[i]]); n++; } val = l / n; }
    else if (m.kind === 'mlp') val = evalMLP(m, valIdx);
    else if (m.kind === 'transformer') val = evalTF(m, nTrain + 1, tokens.length);
    else val = evalRNN(m, nTrain + 1, tokens.length);
  }
  m.val = val ?? j.val;
  models[m.kind] = m;
  return { kind: m.kind, hp: m.hp, params: m.count, val, savedVal: j.val, same, trainedOn: j.trainedOn, step: j.step, sample: sampleText(m, '', .8, 300) };
}

self.onmessage = e => {
  const m = e.data;
  try {
    if (m.cmd === 'data') { if (running) running.stop = true; ftSaved = null; tmask = null; post({ type: 'ready', ...setData(m.text, m.chars, m.label, m.unit) }); }
    else if (m.cmd === 'finetune') { if (running) running.stop = true; setTimeout(() => { try { finetune(m.base, m.hp, m.dialog); } catch (err) { restoreData(); post({ type: 'error', msg: String(err.message || err) }); } }, 0); }
    else if (m.cmd === 'chat') post({ type: 'chat', id: m.id, text: chatReply(m.slot, m.history, m.temp) });
    else if (m.cmd === 'train') {
      if (running) running.stop = true;
      restoreData();
      if (m.kind === 'bigram') trainBigram(); else if (m.kind === 'mlp') trainMLP(m.hp); else if (m.kind === 'transformer') { if (m.hp.device === 'gpu') trainTFGPU(m.hp).catch(err => { running = null; post({ type: 'error', msg: 'GPU: ' + (err && err.message || err) }); }); else trainTF(m.hp); } else trainRNN(m.hp);
    }
    else if (m.cmd === 'stop') { if (running) running.stop = true; }
    else if (m.cmd === 'sample') {
      const model = models[m.kind];
      post({ type: 'sample', id: m.id, kind: m.kind, text: model ? sampleText(model, m.prompt, m.temp, m.len) : null });
    }
    else if (m.cmd === 'gputest') gpuSelfTest().then(r => post({ type: 'gputest', ok: true, ...r }), err => post({ type: 'gputest', ok: false, msg: String(err && err.message || err) }));
    else if (m.cmd === 'export') post({ type: 'export', kind: m.kind, json: exportModel(m.kind) });
    else if (m.cmd === 'import') post({ type: 'imported', ...importModel(m.json) });
  } catch (err) { post({ type: 'error', msg: String(err && err.message || err) }); }
};
