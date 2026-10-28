// Laboratorium języka · wątek treningowy. Liczy w tle, żeby strona nigdy się nie zawieszała.
importScripts('engine.js');

let tokens = null, V = 0, chars = [], nTrain = 0, valIdx = null, nlIdx = 0;
let running = null;                 // aktualny trening: {kind, stop}
const models = {};                  // wytrenowane modele, do próbek i porównań

const post = m => self.postMessage(m);
const randInt = n => Math.floor(Math.random() * n);

function setData(text, ch) {
  chars = ch; V = ch.length;
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
  const w = probs.map(p => Math.pow(Math.max(p, 1e-12), 1 / temp)), s = w.reduce((a, b) => a + b, 0);
  let r = Math.random() * s;
  for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return i; }
  return w.length - 1;
}
function encodePrompt(prompt) { const stoi = new Map(chars.map((c, i) => [c, i])); return [...prompt].map(c => stoi.get(c)).filter(v => v !== undefined); }

// ---------- krok 1: bigram (liczenie par liter) ----------
function trainBigram() {
  const C = new Float64Array(V * V), alpha = .5;      // wygładzanie: para, której nie było, nie dostaje zera
  for (let i = 1; i < nTrain; i++) C[tokens[i - 1] * V + tokens[i]]++;
  const P = new Float64Array(V * V);
  for (let a = 0; a < V; a++) {
    let s = 0; for (let b = 0; b < V; b++) s += C[a * V + b] + alpha;
    for (let b = 0; b < V; b++) P[a * V + b] = (C[a * V + b] + alpha) / s;
  }
  const lossOn = (from, to) => { let l = 0, n = 0; for (let i = Math.max(1, from); i < to; i++) { l -= Math.log(P[tokens[i - 1] * V + tokens[i]]); n++; } return l / n; };
  const model = { kind: 'bigram', P, params: V * V,
    probs(ctx) { const a = ctx.length ? ctx[ctx.length - 1] : nlIdx; return Array.from(P.subarray(a * V, a * V + V)); } };
  models.bigram = model;
  // mapa: najczęstsze litery, wiersz = poprzednia, kolumna = następna
  const freq = new Float64Array(V); for (let i = 0; i < nTrain; i++) freq[tokens[i]]++;
  const top = [...Array(V).keys()].sort((a, b) => freq[b] - freq[a]).slice(0, Math.min(28, V));
  const heat = top.map(a => top.map(b => P[a * V + b]));
  const res = { kind: 'bigram', step: 1, train: lossOn(1, nTrain), val: lossOn(nTrain + 1, tokens.length), params: model.params,
    sample: sampleText(model, '', .9, 240), heat: { labels: top.map(i => chars[i]), m: heat } };
  post({ type: 'progress', ...res }); post({ type: 'done', ...res });
}

// ---------- krok 2: sieć z oknem (Bengio i in., 2003) ----------
function buildMLP(hp) {
  const { ctx, emb, hidden } = hp;
  const E = param([V, emb], 1);
  const W1 = param([ctx * emb, hidden], (5 / 3) / Math.sqrt(ctx * emb));   // skala zgodna z tanh, żeby neurony nie nasycały się od startu
  const b1 = zeros([hidden]);
  const W2 = param([hidden, V], .01);                                       // małe wyjście = na starcie wszystkie litery równie prawdopodobne
  const b2 = zeros([V]);
  const params = [E, W1, b1, W2, b2];
  const forward = idx => {             // idx: Int32Array długości B*ctx
    const B = idx.length / ctx;
    const x = view(gather(E, idx), [B, ctx * emb]);
    const h = tanh(addBias(matmul(x, W1), b1));
    return addBias(matmul(h, W2), b2);
  };
  return { kind: 'mlp', hp, E, params, forward, count: params.reduce((s, p) => s + p.size, 0),
    probs(c) {
      const idx = new Int32Array(ctx);
      for (let j = 0; j < ctx; j++) { const k = c.length - ctx + j; idx[j] = k >= 0 ? c[k] : nlIdx; }
      const lg = forward(idx).d, mx = Math.max(...lg), e = Array.from(lg, v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0);
      return e.map(v => v / s);
    } };
}
function batchFor(positions, ctx) {
  const X = new Int32Array(positions.length * ctx), Y = new Int32Array(positions.length);
  positions.forEach((p, b) => { for (let j = 0; j < ctx; j++) { const k = p - ctx + j; X[b * ctx + j] = k >= 0 ? tokens[k] : nlIdx; } Y[b] = tokens[p]; });
  return { X, Y };
}
function evalMLP(m, idxs) {
  let loss = 0; const ctx = m.hp.ctx, chunk = 500;
  for (let i = 0; i < idxs.length; i += chunk) {
    const part = Array.from(idxs.slice(i, i + chunk)), { X, Y } = batchFor(part, ctx);
    loss += crossEntropy(m.forward(X), Y).d[0] * part.length;
  }
  return loss / idxs.length;
}
function trainMLP(hp) {
  const m = buildMLP(hp), opt = new Adam(m.params, hp.lr), job = { kind: 'mlp', stop: false };
  running = job; models.mlp = m;
  const trainCheck = Int32Array.from({ length: Math.min(3000, nTrain - 1) }, () => 1 + randInt(nTrain - 1));
  let step = 0, ema = null, ticks = 0, lastVal = null;
  const report = (withExtras, withVal = true) => {
    if (withVal) lastVal = evalMLP(m, valIdx);
    const msg = { type: 'progress', kind: 'mlp', step, train: ema, val: lastVal, params: m.count };
    if (withExtras) { msg.sample = sampleText(m, '', .8, 240); msg.emb = { labels: chars, pts: pca2(m.E.d, V, hp.emb) }; }
    post(msg);
  };
  report(true);
  const tick = () => {
    if (job.stop || step >= hp.steps) { finish(); return; }
    const t0 = performance.now();
    while (performance.now() - t0 < 60 && step < hp.steps) {
      const pos = Array.from({ length: hp.batch }, () => 1 + randInt(nTrain - 1));
      const { X, Y } = batchFor(pos, hp.ctx);
      const loss = crossEntropy(m.forward(X), Y);
      opt.zeroGrad(); backward(loss);
      opt.lr = hp.lr * (step > hp.steps * .7 ? .2 : 1);   // pod koniec mniejsze kroki, żeby dopracować szczegóły
      opt.step(); step++;
      ema = ema == null ? loss.d[0] : ema * .97 + loss.d[0] * .03;
    }
    ticks++;   // co ok. 60 ms: błąd treningu; co 5 porcji: walidacja; co 25: próbka tekstu i mapa embeddingów
    report(ticks % 25 === 0, ticks % 5 === 0);
    setTimeout(tick, 0);
  };
  const finish = () => {
    m.trainLoss = evalMLP(m, trainCheck); m.valLoss = evalMLP(m, valIdx);
    post({ type: 'done', kind: 'mlp', step, train: m.trainLoss, val: m.valLoss, params: m.count,
      sample: sampleText(m, '', .8, 300), emb: { labels: chars, pts: pca2(m.E.d, V, hp.emb) }, stopped: job.stop });
    running = null;
  };
  setTimeout(tick, 0);
}

function sampleText(m, prompt, temp, len) {
  const ctx = encodePrompt(prompt), out = [];
  for (let i = 0; i < len; i++) { const n = pick(m.probs(ctx), temp); ctx.push(n); out.push(chars[n]); }
  return out.join('');
}

self.onmessage = e => {
  const m = e.data;
  try {
    if (m.cmd === 'data') { if (running) running.stop = true; setData(m.text, m.chars); post({ type: 'ready', n: tokens.length, train: nTrain }); }
    else if (m.cmd === 'train') {
      if (running) running.stop = true;
      if (m.kind === 'bigram') trainBigram(); else trainMLP(m.hp);
    }
    else if (m.cmd === 'stop') { if (running) running.stop = true; }
    else if (m.cmd === 'sample') {
      const model = models[m.kind];
      post({ type: 'sample', id: m.id, kind: m.kind, text: model ? sampleText(model, m.prompt, m.temp, m.len) : null });
    }
  } catch (err) { post({ type: 'error', msg: String(err && err.message || err) }); }
};
