// Laboratorium obrazów · wątek treningowy. Silnik (autograd, WASM SIMD, WebGPU) wspólny z laboratorium języka.
importScripts('../../lang/js/engine.js', '../../lang/js/wasm.js', '../../lang/js/gpu.js');

const S2 = 256, LV = 16;            // pikseli w obrazku (16×16) i odcieni
let cats = [], train = [], val = [], running = null;
const models = {};
const post = m => self.postMessage(m);
const randInt = n => Math.floor(Math.random() * n);
const now = () => performance.now();

// Dane: dla każdej kategorii tablica pikseli (n × 256, wartości 0..15). 90% do nauki, 10% do sprawdzania.
function setData(list) {
  cats = list.map(c => c.name); train = []; val = [];
  list.forEach((c, ci) => {
    const n = c.pix.length / S2, nv = Math.max(1, Math.round(n * .1));
    for (let i = 0; i < n; i++) (i >= n - nv ? val : train).push({ c: ci, p: c.pix.subarray(i * S2, (i + 1) * S2) });
  });
  for (const k in models) delete models[k];
  return { cats, train: train.length, val: val.length };
}
// Stały zestaw do sprawdzania: po równo z każdej kategorii.
function valSet(S) { const out = [], per = Math.ceil(S / cats.length); cats.forEach((_, ci) => out.push(...val.filter(v => v.c === ci).slice(0, per))); return out.slice(0, S); }

function pick(probs, temp) {
  const w = Array.from(probs, p => Math.pow(Math.max(p, 1e-12), 1 / temp)), s = w.reduce((a, b) => a + b, 0);
  let r = Math.random() * s; for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return i; } return w.length - 1;
}
function softmax(lg) { let mx = -Infinity; for (const v of lg) mx = Math.max(mx, v); const e = Array.from(lg, v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }

// ---------- krok 1: statystyka pikseli ----------
// Dla każdej kategorii i każdego z 256 miejsc liczymy, jak często występuje każdy odcień.
// Każdy piksel losujemy osobno: model nie wie nic o sąsiadach, więc kształty się rozpadają.
function trainStats() {
  const C = cats.length, cnt = new Float64Array(C * S2 * LV), alpha = .5;
  for (const { c, p } of train) for (let i = 0; i < S2; i++) cnt[(c * S2 + i) * LV + p[i]]++;
  const P = new Float32Array(C * S2 * LV);
  for (let k = 0; k < C * S2; k++) { let s = 0; for (let v = 0; v < LV; v++) s += cnt[k * LV + v] + alpha; for (let v = 0; v < LV; v++) P[k * LV + v] = (cnt[k * LV + v] + alpha) / s; }
  const loss = set => { let l = 0; for (const { c, p } of set) for (let i = 0; i < S2; i++) l -= Math.log(P[(c * S2 + i) * LV + p[i]]); return l / (set.length * S2); };
  const m = { kind: 'stats', P, count: C * S2 * LV };
  m.sample = (c, temp) => { const o = new Uint8Array(S2); for (let i = 0; i < S2; i++) o[i] = pick(P.subarray((c * S2 + i) * LV, (c * S2 + i + 1) * LV), temp); return o; };
  models.stats = m;
  const mean = cats.map((_, c) => Array.from({ length: S2 }, (_, i) => { let s = 0; for (let v = 0; v < LV; v++) s += v * P[(c * S2 + i) * LV + v]; return s; }));
  const res = { kind: 'stats', step: 1, train: loss(train), val: loss(val), params: m.count, samples: cats.map((_, c) => Array.from(m.sample(c, 1))), mean };
  post({ type: 'progress', ...res }); post({ type: 'done', ...res });
}

// ---------- krok 2: piksel po pikselu (transformer, jak ImageGPT) ----------
// Wejście: [kategoria, piksel 0, piksel 1, …, piksel 254], cel: [piksel 0, …, piksel 255].
// Na pozycji t sieć widzi kategorię i piksele przed t (wiersz po wierszu) i ocenia 16 możliwych odcieni piksela t.
function buildAR(hp) {
  const { layers: L, dim: d, heads: nh } = hp, T = S2, Vin = LV + cats.length;
  const one = n => { const t = zeros([n]); t.d.fill(1); return t; };
  const Emb = param([Vin, d], .3), P = param([T, d], .1), params = [Emb, P], blocks = [];
  const s = 1 / Math.sqrt(d), sr = s / Math.sqrt(2 * L);
  for (let l = 0; l < L; l++) {
    const b = { g1: one(d), b1: zeros([d]), Wqkv: param([d, 3 * d], s), bqkv: zeros([3 * d]), Wp: param([d, d], sr), bp: zeros([d]),
      g2: one(d), b2: zeros([d]), W1: param([d, 4 * d], s), c1: zeros([4 * d]), W2: param([4 * d, d], sr / 2), c2: zeros([d]) };
    blocks.push(b); params.push(...Object.values(b));
  }
  const gf = one(d), bf = zeros([d]), Wo = param([d, LV], .01), bo = zeros([LV]);
  params.push(gf, bf, Wo, bo);
  const forward = (X, B, Tw) => {
    let x = addPos(gather(Emb, X), P, Tw); const atts = [];
    for (const b of blocks) {
      const at = causalAttention(addBias(matmul(layerNorm(x, b.g1, b.b1), b.Wqkv), b.bqkv), B, Tw, nh); atts.push(at);
      x = add(x, addBias(matmul(at, b.Wp), b.bp));
      x = add(x, addBias(matmul(gelu(addBias(matmul(layerNorm(x, b.g2, b.b2), b.W1), b.c1)), b.W2), b.c2));
    }
    return { logits: addBias(matmul(layerNorm(x, gf, bf), Wo), bo), atts };
  };
  // hp.ctx i chars: ten sam układ wag co transformer tekstowy, więc działa też silnik GPU
  return { kind: 'ar', hp: { ...hp, ctx: T }, chars: Array.from({ length: LV }, (_, i) => String(i)), E: Emb, params, forward, count: params.reduce((a, p) => a + p.size, 0) };
}
function arBatch(set, idxs) {
  const B = idxs.length, X = new Int32Array(B * S2), Y = new Int32Array(B * S2);
  idxs.forEach((k, b) => { const { c, p } = set[k]; X[b * S2] = LV + c; for (let t = 1; t < S2; t++) X[b * S2 + t] = p[t - 1]; Y.set(p, b * S2); });
  return { X, Y };
}
const randBatch = B => arBatch(train, Array.from({ length: B }, () => randInt(train.length)));
// Generowanie z pamięcią kluczy i wartości (KV cache), jak w dużych modelach: każdy narysowany piksel
// liczymy raz, a jego klucze i wartości z każdej warstwy zostają w pamięci. Nowy piksel tylko pyta je uwagą.
// Bez tego każdy krok liczyłby od nowa cały dotychczasowy obrazek (wolniej o ok. 100×). Zwykłe tablice, bez gradientów.
function generate(m, classes, temp, onPart, force) {
  const probs = force ? classes.map(() => []) : null;
  const { layers: L, dim: d, heads: nh } = m.hp, hd = d / nh, sc = 1 / Math.sqrt(hd), P = m.params.map(p => p.d), B = classes.length;
  const Emb = P[0], Pos = P[1], blk = l => P.slice(2 + l * 12, 14 + l * 12), [gf, bf, Wo, bo] = P.slice(2 + L * 12);
  const Kc = Array.from({ length: L }, () => new Float32Array(B * S2 * d)), Vc = Array.from({ length: L }, () => new Float32Array(B * S2 * d));
  const out = classes.map(() => new Uint8Array(S2)), tok = classes.map(c => LV + c);
  const ln = (x, g, b) => { let mu = 0; for (let j = 0; j < d; j++) mu += x[j]; mu /= d; let v = 0; for (let j = 0; j < d; j++) v += (x[j] - mu) ** 2; const r = 1 / Math.sqrt(v / d + 1e-5); return Float32Array.from(x, (z, j) => (z - mu) * r * g[j] + b[j]); };
  const mv = (x, W, bias, n, mOut) => { const o = Float32Array.from(bias); for (let i = 0; i < n; i++) { const xi = x[i]; if (!xi) continue; for (let j = 0; j < mOut; j++) o[j] += xi * W[i * mOut + j]; } return o; };
  const gl = z => .5 * z * (1 + Math.tanh(.7978845608 * (z + .044715 * z * z * z)));
  for (let t = 0; t < S2; t++) {
    for (let b = 0; b < B; b++) {
      let x = new Float32Array(d); for (let j = 0; j < d; j++) x[j] = Emb[tok[b] * d + j] + Pos[t * d + j];
      for (let l = 0; l < L; l++) {
        const [g1, b1, Wqkv, bqkv, Wp, bp, g2, b2, W1, c1, W2, c2] = blk(l), qkv = mv(ln(x, g1, b1), Wqkv, bqkv, d, 3 * d);
        const base = (b * S2 + t) * d; Kc[l].set(qkv.subarray(d, 2 * d), base); Vc[l].set(qkv.subarray(2 * d), base);
        const at = new Float32Array(d), w = new Float32Array(t + 1);
        for (let h = 0; h < nh; h++) {
          let mx = -Infinity;
          for (let j = 0; j <= t; j++) { let s = 0; const kb = (b * S2 + j) * d + h * hd; for (let k = 0; k < hd; k++) s += qkv[h * hd + k] * Kc[l][kb + k]; w[j] = s * sc; if (w[j] > mx) mx = w[j]; }
          let sum = 0; for (let j = 0; j <= t; j++) { w[j] = Math.exp(w[j] - mx); sum += w[j]; }
          for (let j = 0; j <= t; j++) { const p = w[j] / sum, vb = (b * S2 + j) * d + h * hd; for (let k = 0; k < hd; k++) at[h * hd + k] += p * Vc[l][vb + k]; }
        }
        const pr = mv(at, Wp, bp, d, d); for (let j = 0; j < d; j++) x[j] += pr[j];
        const hdn = mv(ln(x, g2, b2), W1, c1, d, 4 * d).map(gl), o2 = mv(hdn, W2, c2, 4 * d, d); for (let j = 0; j < d; j++) x[j] += o2[j];
      }
      const pr = softmax(mv(ln(x, gf, bf), Wo, bo, d, LV)), v = force ? force[b][t] : pick(pr, temp);
      if (probs) probs[b].push(pr);
      out[b][t] = v; tok[b] = v;
    }
    if (onPart && (t % 16 === 15)) onPart(t + 1, out);
  }
  return force ? probs : out;
}
// Wersja bez pamięci kluczy (wolna), zostawiona do porównania szybkości i sprawdzenia zgodności.
function generateSlow(m, classes, temp, onPart) {
  const B = classes.length, seqs = classes.map(c => [LV + c]), out = classes.map(() => new Uint8Array(S2));
  for (let t = 0; t < S2; t++) {
    const len = t + 1, X = new Int32Array(B * len);
    seqs.forEach((s, b) => X.set(s, b * len));
    const lg = m.forward(X, B, len).logits.d;
    for (let b = 0; b < B; b++) {
      const v = pick(softmax(lg.subarray(((b * len) + len - 1) * LV, ((b * len) + len) * LV)), temp);
      out[b][t] = v; seqs[b].push(v);
    }
    if (onPart && (t % 16 === 15)) onPart(t + 1, out);
  }
  return out;
}
// Uwaga na obrazku: dla jednego obrazka sprawdzającego wagi uwagi wszystkich warstw i głów.
function attnProbe(m) {
  const v = val[randInt(val.length)], { X } = arBatch([v], [0]), { atts } = m.forward(X, 1, S2);
  return { c: v.c, pix: Array.from(v.p), L: atts.length, nh: m.hp.heads, att: atts.map(a => a.att) };
}
function evalAR(m, set) {
  const { X, Y } = arBatch(set, set.map((_, i) => i));
  return crossEntropy(m.forward(X, set.length, S2).logits, Y).d[0];
}
// Widok neuronów: próbka wag między pokazanymi kółkami (12 z d, 16 z 4d, wybrane wejścia i 16 wyjść).
const spread = (n, k) => Array.from({ length: Math.min(n, k) }, (_, i) => Math.round(i * (n - 1) / Math.max(1, Math.min(n, k) - 1)));
function weightViz(m) {
  const { layers: L, dim: d } = m.hp, P = m.params.map(p => p.d), di = spread(d, 12), hi = spread(4 * d, 16);
  const ti = [0, 3, 6, 9, 12, 15, ...cats.map((_, i) => LV + i).slice(0, 6)];
  const sub = (W, cols, rows, cs) => rows.map(r => cs.map(c => W[r * cols + c]));
  const blocks = Array.from({ length: L }, (_, l) => ({ w1: sub(P[2 + l * 12 + 8], 4 * d, di, hi), w2: sub(P[2 + l * 12 + 10], d, hi, di) }));
  return { tok: ti.map(t => t < LV ? `odcień ${t}` : cats[t - LV]), emb: sub(P[0], d, ti, di), blocks, wo: sub(P[2 + L * 12 + 2], LV, di, [...Array(LV).keys()]) };
}
function extrasAR(m) {
  const t0 = now(), cls = cats.map((_, i) => i).slice(0, 6), imgs = generate(m, cls, 1);
  return { samples: imgs.map(a => Array.from(a)), sampleClasses: cls, attn: attnProbe(m), genMs: now() - t0, wviz: weightViz(m) };
}

function trainAR(hp) {
  try { setBackend(hp.device === 'cpu' ? 'js' : 'wasm'); } catch (e) { setBackend('js'); }
  const m = buildAR(hp), opt = new Adam(m.params, hp.lr), job = { stop: false }, vs = valSet(32), ts = train.slice(0, 32);
  running = job; models.ar = m;
  const dev = hp.device === 'cpu' ? 'JS' : 'WASM SIMD';
  let step = 0, ema = null, lastVal = evalAR(m, vs), best = { val: lastVal, step: 0, w: m.params.map(p => p.d.slice()) }, tVal = now(), tEx = now(), ms = null;
  const base = () => ({ kind: 'ar', step, train: ema, val: lastVal, params: m.count, bestStep: best.step, ms, device: dev });
  post({ type: 'progress', ...base(), wviz: weightViz(m) });
  const tick = () => {
    if (job.stop || step >= hp.steps) return finish();
    const t0 = now(), s0 = step;
    while (now() - t0 < 80 && step < hp.steps) {
      const { X, Y } = randBatch(hp.batch), loss = crossEntropy(m.forward(X, hp.batch, S2).logits, Y);
      opt.zeroGrad(); backward(loss); opt.clip(1);
      opt.lr = hp.lr * (step > hp.steps * .7 ? .2 : 1); opt.step(); step++;
      ema = ema == null ? loss.d[0] : ema * .95 + loss.d[0] * .05;
    }
    ms = (now() - t0) / (step - s0);
    const msg = { type: 'progress', ...base() };
    if (now() - tVal > 3000) { tVal = now(); lastVal = evalAR(m, vs); if (lastVal < best.val - 1e-4) best = { val: lastVal, step, w: m.params.map(p => p.d.slice()) }; Object.assign(msg, base()); }
    if (now() - tEx > 20000) { tEx = now(); Object.assign(msg, extrasAR(m)); }
    post(msg); setTimeout(tick, 0);
  };
  const finish = () => {
    let v = evalAR(m, vs), restored = false;
    if (v < best.val - 1e-4) best = { val: v, step, w: null }; else if (v > best.val + .01) { m.params.forEach((p, i) => p.d.set(best.w[i])); v = best.val; restored = true; }
    lastVal = v; running = null;
    post({ type: 'done', ...base(), val: v, train: evalAR(m, ts), restored, ...extrasAR(m) });
  };
  setTimeout(tick, 0);
}

// ---------- krok 2 na karcie graficznej ----------
let gpuEng = null;
// Zgodność: ten sam start i te same rysunki, jeden krok nauki na CPU i na GPU.
async function gpuSelfTest() {
  if (!gpuEng || gpuEng.lost) gpuEng = await GPUEngine.create();
  setBackend('wasm');
  const hp = { layers: 2, dim: 32, heads: 4 }, mC = buildAR(hp), mG = buildAR(hp), { X, Y } = randBatch(2);
  mG.params.forEach((p, i) => p.d.set(mC.params[i].d));
  const opt = new Adam(mC.params, .003), loss = crossEntropy(mC.forward(X, 2, S2).logits, Y);
  opt.zeroGrad(); backward(loss); opt.clip(1); opt.step();
  gpuEng.dev.pushErrorScope('validation');
  const g = new GPUTransformer(gpuEng, mG, 2, 1); g.steps([{ X, Y }], .003, 1);
  const err = await gpuEng.dev.popErrorScope(); if (err) throw new Error('WebGPU: ' + err.message);
  const lossG = await g.lastLoss(); await g.pull();
  let diff = 0; mG.params.forEach((p, i) => { for (let j = 0; j < p.size; j++) diff = Math.max(diff, Math.abs(p.d[j] - mC.params[i].d[j])); });
  return { name: gpuEng.name, lossC: loss.d[0], lossG, diff };
}
async function trainARGPU(hp) {
  if (!gpuEng || gpuEng.lost) gpuEng = await GPUEngine.create();
  setBackend('wasm');
  const m = buildAR(hp), job = { stop: false }, vs = valSet(32), ts = train.slice(0, 32), S = vs.length;
  running = job; models.ar = m;
  const g = new GPUTransformer(gpuEng, m, hp.batch, S), vXY = arBatch(vs, vs.map((_, i) => i)), tXY = arBatch(ts, ts.map((_, i) => i));
  const evalLoss = async ({ X, Y }) => { const r = await g.evalRows(X, Y); let l = 0; for (const v of r) l += v; return l / r.length; };
  let step = 0, ema = null, lastVal = await evalLoss(vXY), best = { val: lastVal, step: 0 }, tVal = now(), tEx = now(), k = 1, ms = null;
  g.snapshot();
  const dev = 'GPU · ' + gpuEng.name;
  const base = () => ({ kind: 'ar', step, train: ema, val: lastVal, params: m.count, bestStep: best.step, ms, device: dev });
  post({ type: 'progress', ...base(), wviz: weightViz(m) });
  while (!job.stop && step < hp.steps) {
    const n = Math.min(k, hp.steps - step), t0 = now();
    g.steps(Array.from({ length: n }, () => randBatch(hp.batch)), hp.lr * (step > hp.steps * .7 ? .2 : 1), 1);
    const L = await g.lastLoss(); step += n;
    ms = (now() - t0) / n; k = Math.max(1, Math.min(64, Math.round(200 / ms)));
    ema = ema == null ? L : ema * Math.pow(.95, n) + L * (1 - Math.pow(.95, n));
    const msg = { type: 'progress' };
    if (now() - tVal > 2000) { tVal = now(); lastVal = await evalLoss(vXY); if (lastVal < best.val - 1e-4) { best = { val: lastVal, step }; g.snapshot(); } }
    if (now() - tEx > 20000) { tEx = now(); await g.pull(); Object.assign(msg, extrasAR(m)); }
    post({ ...msg, ...base() });
    await new Promise(r => setTimeout(r, 0));
  }
  let v = await evalLoss(vXY), restored = false;
  if (v < best.val - 1e-4) best = { val: v, step }; else if (v > best.val + .01) { g.restore(); v = best.val; restored = true; }
  await g.pull(); lastVal = v; running = null;
  post({ type: 'done', ...base(), val: v, train: await evalLoss(tXY), restored, ...extrasAR(m) });
}

self.onmessage = e => {
  const m = e.data;
  try {
    if (m.cmd === 'data') { if (running) running.stop = true; post({ type: 'ready', ...setData(m.cats) }); }
    else if (m.cmd === 'train') {
      if (running) running.stop = true;
      if (m.kind === 'stats') trainStats();
      else if (m.hp.device === 'gpu') trainARGPU(m.hp).catch(err => { running = null; post({ type: 'error', msg: 'GPU: ' + (err.message || err) }); });
      else trainAR(m.hp);
    }
    else if (m.cmd === 'stop') { if (running) running.stop = true; }
    else if (m.cmd === 'gputest') gpuSelfTest().then(r => post({ type: 'gputest', ok: true, ...r }), err => post({ type: 'gputest', ok: false, msg: String(err.message || err) }));
    else if (m.cmd === 'generate') {
      const md = models[m.kind]; if (!md) throw new Error('Najpierw naucz model.');
      if (md.kind === 'stats') post({ type: 'gen', id: m.id, done: true, n: S2, imgs: m.classes.map(c => Array.from(md.sample(c, m.temp))) });
      else { const out = generate(md, m.classes, m.temp, (n, o) => post({ type: 'gen', id: m.id, n, imgs: o.map(a => Array.from(a)) })); post({ type: 'gen', id: m.id, done: true, n: S2, imgs: out.map(a => Array.from(a)) }); }
    }
  } catch (err) { post({ type: 'error', msg: String(err && err.message || err) }); }
};
