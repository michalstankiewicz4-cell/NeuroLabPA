// Laboratorium języka · wątek treningowy. Liczy w tle, żeby strona nigdy się nie zawieszała.
importScripts('engine.js');

let tokens = null, V = 0, chars = [], nTrain = 0, valIdx = null, nlIdx = 0, dataLabel = '';
let running = null;                 // aktualny trening: {kind, stop}
const models = {};                  // wytrenowane modele, do próbek, zapisu i porównań

const post = m => self.postMessage(m);
const randInt = n => Math.floor(Math.random() * n);
const now = () => performance.now();

function setData(text, ch, label) {
  chars = ch; V = ch.length; dataLabel = label || '';
  const stoi = new Map(ch.map((c, i) => [c, i]));
  tokens = new Int32Array(text.length);
  for (let i = 0; i < text.length; i++) tokens[i] = stoi.get(text[i]) ?? 0;
  nTrain = Math.floor(tokens.length * .9);           // ostatnie 10% tekstu zostaje do sprawdzania (walidacja)
  const nv = Math.min(3000, tokens.length - nTrain - 1);
  valIdx = Int32Array.from({ length: nv }, () => nTrain + 1 + randInt(tokens.length - nTrain - 1));
  nlIdx = stoi.has('\n') ? stoi.get('\n') : 0;
  for (const k in models) delete models[k];
}

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
function sampleText(m, prompt, temp, len) {
  const stoi = new Map(m.chars.map((c, i) => [c, i]));
  const ctx = [...prompt].map(c => stoi.get(c)).filter(v => v !== undefined), out = [], next = m.stepper();
  for (let i = 0; i < len; i++) { const n = pick(next(ctx), temp); ctx.push(n); out.push(m.chars[n]); }
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
  const model = bigramModel(P, chars);
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
  positions.forEach((p, b) => { for (let j = 0; j < ctx; j++) { const k = p - ctx + j; X[b * ctx + j] = k >= 0 ? tokens[k] : nlIdx; } Y[b] = tokens[p]; });
  return { X, Y };
}
function evalMLP(m, idxs) {
  let loss = 0; const ctx = m.hp.ctx, chunk = 500;
  for (let i = 0; i < idxs.length; i += chunk) {
    const part = Array.from(idxs.slice(i, i + chunk)), { X, Y } = mlpBatch(part, ctx);
    loss += crossEntropy(m.forward(X), Y).d[0] * part.length;
  }
  return loss / idxs.length;
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
    for (let t = 0; t < count; t++) Y[t * S + i] = tokens[s + warm + t + 1];
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
    const c = m.chars[X[t]];
    pos = isL(c) ? pos + 1 : 0;
    if (c === '„' || c === '«') q = 1; else if (c === '”' || c === '»') q = 0; else if (c === '"') q = 1 - q;
    if (c === '(') par++; else if (c === ')') par = Math.max(0, par - 1);
    if (c === '\n') { q = 0; par = 0; }
    feats['litera (nie odstęp)'].push(isL(c) ? 1 : 0); feats['pozycja w słowie'].push(Math.min(pos, 10));
    feats['w cudzysłowie'].push(q); feats['w nawiasie'].push(par > 0 ? 1 : 0); feats['samogłoska'].push('aąeęioóuyAĄEĘIOÓUY'.includes(c) ? 1 : 0);
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
  return { text: Array.from(X.subarray(s0), i => m.chars[i]).join(''), H, acts: acts.slice(s0 * H), found };
}
function rnnBatch(B, Tn) {
  const X = new Int32Array(Tn * B), Y = new Int32Array(Tn * B);
  for (let i = 0; i < B; i++) { const s = randInt(nTrain - Tn - 1); for (let t = 0; t < Tn; t++) { X[t * B + i] = tokens[s + t]; Y[t * B + i] = tokens[s + t + 1]; } }
  return { X, Y };
}

// ---------- wspólna pętla nauki (krok 2 i 3) ----------
// Pamięta najlepszą wersję sieci: gdy błąd sprawdzający zaczyna rosnąć (uczenie na pamięć),
// na koniec wracamy do wag z najlepszego momentu.
function trainLoop(m, hp, fns) {
  const opt = new Adam(m.params, hp.lr), job = { kind: m.kind, stop: false };
  running = job; models[m.kind] = m; m.label = dataLabel;
  let step = 0, ema = null, lastVal = null, best = null, tVal = 0, tExtra = 0;
  const snap = () => m.params.map(p => p.d.slice());
  const report = (extras, withVal) => {
    if (withVal) {
      lastVal = fns.val();
      if (!best || lastVal < best.val - 1e-4) best = { val: lastVal, step, w: snap() };
    }
    const msg = { type: 'progress', kind: m.kind, step, train: ema, val: lastVal, params: m.count, bestStep: best && best.step, bestVal: best && best.val };
    if (extras) Object.assign(msg, { sample: sampleText(m, '', .8, 240) }, fns.extras());
    post(msg);
  };
  report(true, true);
  const tick = () => {
    if (job.stop || step >= hp.steps) { finish(); return; }
    const t0 = now();
    while (now() - t0 < 60 && step < hp.steps) {
      const loss = fns.loss();
      opt.zeroGrad(); backward(loss);
      if (fns.clip) opt.clip(fns.clip);
      opt.lr = hp.lr * (step > hp.steps * .7 ? .2 : 1);   // pod koniec mniejsze kroki, żeby dopracować szczegóły
      opt.step(); step++;
      ema = ema == null ? loss.d[0] : ema * .97 + loss.d[0] * .03;
    }
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
    post({ type: 'done', kind: m.kind, step, train: fns.train(), val, params: m.count, bestStep: best.step, bestVal: best.val, restored,
      sample: sampleText(m, '', .8, 300), stopped: job.stop, hp, ...fns.extras() });
    running = null;
  };
  setTimeout(tick, 0);
}

function trainMLP(hp) {
  const m = buildMLP(hp, chars);
  const trainCheck = Int32Array.from({ length: Math.min(3000, nTrain - 1) }, () => 1 + randInt(nTrain - 1));
  trainLoop(m, hp, {
    loss: () => { const pos = Array.from({ length: hp.batch }, () => 1 + randInt(nTrain - 1)), { X, Y } = mlpBatch(pos, hp.ctx); return crossEntropy(m.forward(X), Y); },
    val: () => evalMLP(m, valIdx), train: () => evalMLP(m, trainCheck), valMs: 400,
    extras: () => ({ emb: { labels: chars, pts: pca2(m.E.d, V, hp.emb) } }),
  });
}
function trainRNN(hp) {
  const m = buildRNN(hp, chars);
  trainLoop(m, hp, {
    loss: () => { const { X, Y } = rnnBatch(hp.batch, hp.seq); return crossEntropy(m.out(m.run(X, hp.batch)), Y); },
    val: () => evalRNN(m, nTrain + 1, tokens.length), train: () => evalRNN(m, 1, nTrain), valMs: 2500, clip: 1,
    extras: () => ({ reach: gradReach(m), neurons: neuronProbe(m), emb: { labels: chars, pts: pca2(m.E.d, V, hp.emb) } }),
  });
}

// ---------- zapis i wczytanie modelu ----------
const b64 = f => { const u = new Uint8Array(f.buffer, f.byteOffset, f.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
const unb64 = s => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); };
function exportModel(kind) {
  const m = models[kind];
  if (!m) throw new Error('Brak wytrenowanego modelu tego rodzaju.');
  const tensors = m.kind === 'bigram' ? [m.P] : m.params.map(p => p.d);
  return { format: 'neurolab-lang-model', version: 1, kind: m.kind, hp: m.hp, chars: m.chars, trainedOn: m.label || '',
    val: m.val ?? null, step: m.step ?? null, params: m.count, savedAt: new Date().toISOString(), tensors: tensors.map(b64) };
}
function importModel(j) {
  if (!j || j.format !== 'neurolab-lang-model') throw new Error('To nie jest plik modelu z Laboratorium języka.');
  const t = j.tensors.map(unb64); let m;
  if (j.kind === 'bigram') m = bigramModel(t[0], j.chars);
  else {
    m = j.kind === 'mlp' ? buildMLP(j.hp, j.chars) : buildRNN(j.hp, j.chars);
    if (t.length !== m.params.length || t.some((a, i) => a.length !== m.params[i].size)) throw new Error('Plik modelu jest niekompletny albo z innej wersji.');
    m.params.forEach((p, i) => p.d.set(t[i]));
  }
  m.label = j.trainedOn; m.step = j.step;
  const same = j.chars.length === chars.length && j.chars.every((c, i) => c === chars[i]);
  let val = null;
  if (same) {
    if (m.kind === 'bigram') { let l = 0, n = 0; for (let i = nTrain + 1; i < tokens.length; i++) { l -= Math.log(m.P[tokens[i - 1] * V + tokens[i]]); n++; } val = l / n; }
    else if (m.kind === 'mlp') val = evalMLP(m, valIdx);
    else val = evalRNN(m, nTrain + 1, tokens.length);
  }
  m.val = val ?? j.val;
  models[m.kind] = m;
  return { kind: m.kind, hp: m.hp, params: m.count, val, savedVal: j.val, same, trainedOn: j.trainedOn, step: j.step, sample: sampleText(m, '', .8, 300) };
}

self.onmessage = e => {
  const m = e.data;
  try {
    if (m.cmd === 'data') { if (running) running.stop = true; setData(m.text, m.chars, m.label); post({ type: 'ready', n: tokens.length, train: nTrain }); }
    else if (m.cmd === 'train') {
      if (running) running.stop = true;
      if (m.kind === 'bigram') trainBigram(); else if (m.kind === 'mlp') trainMLP(m.hp); else trainRNN(m.hp);
    }
    else if (m.cmd === 'stop') { if (running) running.stop = true; }
    else if (m.cmd === 'sample') {
      const model = models[m.kind];
      post({ type: 'sample', id: m.id, kind: m.kind, text: model ? sampleText(model, m.prompt, m.temp, m.len) : null });
    }
    else if (m.cmd === 'export') post({ type: 'export', kind: m.kind, json: exportModel(m.kind) });
    else if (m.cmd === 'import') post({ type: 'imported', ...importModel(m.json) });
  } catch (err) { post({ type: 'error', msg: String(err && err.message || err) }); }
};
