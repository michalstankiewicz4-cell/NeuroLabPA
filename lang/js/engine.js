// Laboratorium języka · silnik: tensory z automatycznym liczeniem gradientów (autograd) i optymalizator Adam.
//
// Każda operacja tworzy nowy tensor i zapamiętuje, z czego powstała (`prev`) oraz jak przekazać
// gradient wstecz (`back`). backward() przechodzi ten graf od końca i stosuje regułę łańcuchową.
// Wszystko jest jawne i krótkie, żeby dało się przeczytać, co sieć faktycznie liczy.

class T {
  constructor(data, shape, prev = [], back = null) {
    this.d = data;          // Float32Array z wartościami (wiersz po wierszu)
    this.s = shape;         // np. [wiersze, kolumny]
    this.g = null;          // gradient tej samej wielkości, tworzony przy backward()
    this.prev = prev;
    this.back = back;
  }
  get size() { return this.d.length; }
}

// Parametr: liczby losowe z rozkładu normalnego, przeskalowane (inicjalizacja ma znaczenie, patrz opis kroku 2).
function randn() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function param(shape, scale) {
  const n = shape.reduce((a, b) => a * b, 1), d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = randn() * scale;
  const t = new T(d, shape); t.isParam = true; return t;
}
function zeros(shape) { const t = new T(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape); t.isParam = true; return t; }

// [n,k] × [k,m] → [n,m]
function matmul(a, b) {
  const [n, k] = a.s, m = b.s[1], A = a.d, B = b.d, O = new Float32Array(n * m);
  for (let i = 0; i < n; i++) {
    const oi = i * m, ai = i * k;
    for (let p = 0; p < k; p++) { const av = A[ai + p]; if (av === 0) continue; const bp = p * m; for (let j = 0; j < m; j++) O[oi + j] += av * B[bp + j]; }
  }
  const out = new T(O, [n, m], [a, b], () => {
    const G = out.g, AG = a.g, BG = b.g;
    for (let i = 0; i < n; i++) {
      const oi = i * m, ai = i * k;
      for (let p = 0; p < k; p++) {
        let s = 0; const bp = p * m, av = A[ai + p];
        for (let j = 0; j < m; j++) { const g = G[oi + j]; s += g * B[bp + j]; BG[bp + j] += av * g; }
        AG[ai + p] += s;
      }
    }
  });
  return out;
}

// [n,m] + wektor [m] dodawany do każdego wiersza
function addBias(a, b) {
  const [n, m] = a.s, O = new Float32Array(a.d);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) O[i * m + j] += b.d[j];
  const out = new T(O, a.s, [a, b], () => {
    for (let i = 0; i < n * m; i++) a.g[i] += out.g[i];
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) b.g[j] += out.g[i * m + j];
  });
  return out;
}

function tanh(a) {
  const O = new Float32Array(a.size);
  for (let i = 0; i < O.length; i++) O[i] = Math.tanh(a.d[i]);
  const out = new T(O, a.s, [a], () => { for (let i = 0; i < O.length; i++) a.g[i] += (1 - O[i] * O[i]) * out.g[i]; });
  return out;
}

function sigmoid(a) {
  const O = new Float32Array(a.size);
  for (let i = 0; i < O.length; i++) O[i] = 1 / (1 + Math.exp(-a.d[i]));
  const out = new T(O, a.s, [a], () => { for (let i = 0; i < O.length; i++) a.g[i] += O[i] * (1 - O[i]) * out.g[i]; });
  return out;
}

// Działania po elementach na tensorach tego samego kształtu (potrzebne bramkom GRU).
function add(a, b) {
  const O = new Float32Array(a.size); for (let i = 0; i < O.length; i++) O[i] = a.d[i] + b.d[i];
  const out = new T(O, a.s, [a, b], () => { for (let i = 0; i < O.length; i++) { a.g[i] += out.g[i]; b.g[i] += out.g[i]; } });
  return out;
}
function sub(a, b) {
  const O = new Float32Array(a.size); for (let i = 0; i < O.length; i++) O[i] = a.d[i] - b.d[i];
  const out = new T(O, a.s, [a, b], () => { for (let i = 0; i < O.length; i++) { a.g[i] += out.g[i]; b.g[i] -= out.g[i]; } });
  return out;
}
function mul(a, b) {
  const O = new Float32Array(a.size); for (let i = 0; i < O.length; i++) O[i] = a.d[i] * b.d[i];
  const out = new T(O, a.s, [a, b], () => { for (let i = 0; i < O.length; i++) { a.g[i] += out.g[i] * b.d[i]; b.g[i] += out.g[i] * a.d[i]; } });
  return out;
}

// Wycinki i sklejanie: wiersze [start, start+n) albo kolumny [c0, c1) macierzy [n,m].
function sliceRows(a, start, n) {
  const m = a.s[1], out = new T(a.d.slice(start * m, (start + n) * m), [n, m], [a], () => {
    for (let i = 0; i < n * m; i++) a.g[start * m + i] += out.g[i];
  });
  return out;
}
function sliceCols(a, c0, c1) {
  const [n, m] = a.s, w = c1 - c0, O = new Float32Array(n * w);
  for (let i = 0; i < n; i++) O.set(a.d.subarray(i * m + c0, i * m + c1), i * w);
  const out = new T(O, [n, w], [a], () => { for (let i = 0; i < n; i++) for (let j = 0; j < w; j++) a.g[i * m + c0 + j] += out.g[i * w + j]; });
  return out;
}
function concatRows(list) {
  const m = list[0].s[1], n = list.reduce((s, t) => s + t.s[0], 0), O = new Float32Array(n * m);
  let off = 0; for (const t of list) { O.set(t.d, off); off += t.size; }
  const out = new T(O, [n, m], list, () => { let o = 0; for (const t of list) { for (let i = 0; i < t.size; i++) t.g[i] += out.g[o + i]; o += t.size; } });
  return out;
}

// Wybiera wiersze macierzy E według indeksów: tak litery zamieniają się w wektory (embeddingi).
function gather(E, idx) {
  const d = E.s[1], L = idx.length, O = new Float32Array(L * d);
  for (let i = 0; i < L; i++) O.set(E.d.subarray(idx[i] * d, idx[i] * d + d), i * d);
  const out = new T(O, [L, d], [E], () => {
    for (let i = 0; i < L; i++) { const r = idx[i] * d; for (let j = 0; j < d; j++) E.g[r + j] += out.g[i * d + j]; }
  });
  return out;
}

// Ten sam blok liczb, inny kształt (np. 8 embeddingów po 16 liczb = jeden wektor 128 liczb).
function view(a, shape) {
  const out = new T(a.d, shape, [a], () => { for (let i = 0; i < a.size; i++) a.g[i] += out.g[i]; });
  return out;
}

// Softmax + entropia krzyżowa naraz: średni błąd przewidywania następnej litery.
// Błąd = −ln(prawdopodobieństwo, jakie sieć dała prawdziwej literze).
function crossEntropy(logits, targets) {
  const [n, V] = logits.s, P = new Float32Array(n * V); let loss = 0;
  for (let i = 0; i < n; i++) {
    const o = i * V; let mx = -Infinity;
    for (let j = 0; j < V; j++) mx = Math.max(mx, logits.d[o + j]);
    let sum = 0;
    for (let j = 0; j < V; j++) { const e = Math.exp(logits.d[o + j] - mx); P[o + j] = e; sum += e; }
    for (let j = 0; j < V; j++) P[o + j] /= sum;
    loss -= Math.log(P[o + targets[i]] + 1e-12);
  }
  const out = new T(new Float32Array([loss / n]), [1], [logits], () => {
    const g = out.g[0] / n;
    for (let i = 0; i < n; i++) { const o = i * V; for (let j = 0; j < V; j++) logits.g[o + j] += g * (P[o + j] - (j === targets[i] ? 1 : 0)); }
  });
  out.probs = P;
  return out;
}

// ---------- komórki sieci rekurencyjnych: jeden krok czytania (jedna litera) ----------
// xw: wkład bieżącej litery (embedding × W + bias), h: pamięć po poprzedniej literze, U: wagi pamięci.

// Zwykła RNN (Elman 1990): nowa pamięć = tanh(litera + stara pamięć × U).
function rnnCell(xw, h, U) { return tanh(add(xw, matmul(h, U))); }

// GRU (Cho i in. 2014). Dwie bramki z wartościami 0..1:
//   r („reset”): ile starej pamięci użyć przy liczeniu kandydata n,
//   z („update”): ile starej pamięci zostawić bez zmian. Gdy z≈1, pamięć przechodzi dalej nietknięta,
//   więc gradient też przechodzi bez osłabiania; stąd dłuższa pamięć niż w zwykłej RNN.
function gruCell(xw, h, U, bhn) {
  const H = h.s[1], hu = matmul(h, U);
  const r = sigmoid(add(sliceCols(xw, 0, H), sliceCols(hu, 0, H)));
  const z = sigmoid(add(sliceCols(xw, H, 2 * H), sliceCols(hu, H, 2 * H)));
  const n = tanh(add(sliceCols(xw, 2 * H, 3 * H), mul(r, addBias(sliceCols(hu, 2 * H, 3 * H), bhn))));
  return add(n, mul(z, sub(h, n)));          // = (1−z)·n + z·h
}

// Przejście wstecz: kolejność topologiczna, potem reguła łańcuchowa od wyniku do parametrów.
function backward(root) {
  const order = [], seen = new Set();
  (function visit(t) { if (seen.has(t)) return; seen.add(t); t.prev.forEach(visit); order.push(t); })(root);
  for (const t of order) { if (!t.g || !t.isParam) t.g = new Float32Array(t.size); }
  root.g[0] = 1;
  for (let i = order.length - 1; i >= 0; i--) if (order[i].back) order[i].back();
}

// Adam: każdy parametr ma własne, uśrednione tempo kroku. Domyślny wybór, bo wybacza złe tempo nauki.
class Adam {
  constructor(params, lr = 3e-3, b1 = .9, b2 = .999, eps = 1e-8) {
    Object.assign(this, { params, lr, b1, b2, eps, t: 0 });
    this.m = params.map(p => new Float32Array(p.size)); this.v = params.map(p => new Float32Array(p.size));
    params.forEach(p => { p.g = new Float32Array(p.size); });
  }
  zeroGrad() { this.params.forEach(p => p.g.fill(0)); }
  // Obcinanie gradientu: gdy jego długość (norma) przekracza `max`, skalujemy go w dół.
  // W sieciach rekurencyjnych gradient potrafi nagle urosnąć tysiące razy („eksplodujący gradient”).
  clip(max) {
    let s = 0; this.params.forEach(p => { for (let i = 0; i < p.size; i++) s += p.g[i] * p.g[i]; });
    const norm = Math.sqrt(s);
    if (norm > max) { const k = max / norm; this.params.forEach(p => { for (let i = 0; i < p.size; i++) p.g[i] *= k; }); }
    return norm;
  }
  step() {
    this.t++;
    const { b1, b2, eps, lr } = this, c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    this.params.forEach((p, k) => {
      const m = this.m[k], v = this.v[k], g = p.g, d = p.d;
      for (let i = 0; i < p.size; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        d[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
      }
    });
  }
}

// Dwie główne składowe (PCA) macierzy wierszy: rzut embeddingów liter na płaszczyznę.
function pca2(M, rows, cols) {
  const mean = new Float64Array(cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) mean[j] += M[i * cols + j] / rows;
  const X = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) X[i * cols + j] = M[i * cols + j] - mean[j];
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let v = new Float64Array(cols).map((_, j) => Math.sin(j * 1.7 + c + 1));
    for (let it = 0; it < 60; it++) {
      const w = new Float64Array(cols);
      for (let i = 0; i < rows; i++) { let s = 0; for (let j = 0; j < cols; j++) s += X[i * cols + j] * v[j]; for (let j = 0; j < cols; j++) w[j] += s * X[i * cols + j]; }
      comps.forEach(u => { let dot = 0; for (let j = 0; j < cols; j++) dot += w[j] * u[j]; for (let j = 0; j < cols; j++) w[j] -= dot * u[j]; });
      const nrm = Math.hypot(...w) || 1; v = w.map(x => x / nrm);
    }
    comps.push(v);
  }
  const pts = [];
  for (let i = 0; i < rows; i++) { const p = [0, 0]; for (let c = 0; c < 2; c++) for (let j = 0; j < cols; j++) p[c] += X[i * cols + j] * comps[c][j]; pts.push(p); }
  return pts;
}

if (typeof module !== 'undefined') module.exports = { T, param, zeros, matmul, addBias, tanh, sigmoid, add, sub, mul, sliceRows, sliceCols, concatRows, rnnCell, gruCell, gather, view, crossEntropy, backward, Adam, pca2 };
