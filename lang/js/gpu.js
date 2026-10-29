// Laboratorium języka · silnik na karcie graficznej (WebGPU) dla transformera.
//
// Te same działania co w engine.js, ale jako małe programy (shadery WGSL) uruchamiane na karcie.
// Kluczowa różnica: dane nie wracają do procesora między działaniami. Cały krok nauki (przejście
// w przód, wstecz, przycięcie gradientu i Adam) zostaje nagrany raz jako lista zadań i potem tylko
// odtwarzany. Do procesora wraca wyłącznie błąd, a wagi raz na kilka sekund (do próbek i wykresów).

const WG = 64;
const HDR = 'struct P { n: u32, k: u32, m: u32, x: u32, y: u32, z: u32, f0: f32, f1: f32 };\n@group(0) @binding(0) var<uniform> p: P;\n';
const KERNELS = {
  // C[n,m] (+)= op(A)[n,k] · op(B)[k,m]; x=1: A zapisane transponowane, y=1: B transponowane, z=1: dodaj do C
  matmul: `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
var<workgroup> tA: array<f32, 256>;
var<workgroup> tB: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let row = w.y * 16u + l.y; let col = w.x * 16u + l.x;
  var s = 0.0;
  let tiles = (p.k + 15u) / 16u;
  for (var t = 0u; t < tiles; t++) {
    let ac = t * 16u + l.x;
    var av = 0.0;
    if (row < p.n && ac < p.k) { if (p.x == 1u) { av = A[ac * p.n + row]; } else { av = A[row * p.k + ac]; } }
    tA[l.y * 16u + l.x] = av;
    let br = t * 16u + l.y;
    var bv = 0.0;
    if (br < p.k && col < p.m) { if (p.y == 1u) { bv = B[col * p.k + br]; } else { bv = B[br * p.m + col]; } }
    tB[l.y * 16u + l.x] = bv;
    workgroupBarrier();
    for (var q = 0u; q < 16u; q++) { s += tA[l.y * 16u + q] * tB[q * 16u + l.x]; }
    workgroupBarrier();
  }
  if (row < p.n && col < p.m) { if (p.z == 1u) { C[row * p.m + col] += s; } else { C[row * p.m + col] = s; } }
}`,
  // out = a + bias (bias co m elementów)
  addBias: `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) { let i = g.x; if (i >= p.n) { return; } o[i] = a[i] + b[i % p.m]; }`,
  add: `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) { let i = g.x; if (i >= p.n) { return; } o[i] = a[i] + b[i]; }`,
  // dst += src
  axpy: `
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) { let i = g.x; if (i >= p.n) { return; } d[i] += s[i]; }`,
  // db[j] += suma po wierszach g[i, j]   (n wierszy, m kolumn)
  colsum: `
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) { let j = g.x; if (j >= p.m) { return; } var t = 0.0; for (var i = 0u; i < p.n; i++) { t += s[i * p.m + j]; } d[j] += t; }`,
  gelu: `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) { let i = g.x; if (i >= p.n) { return; } let x = a[i]; o[i] = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }`,
  geluBack: `
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> go: array<f32>;
@group(0) @binding(3) var<storage, read_write> ga: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let x = a[i]; let c = 0.7978845608; let t = tanh(c * (x + 0.044715 * x * x * x));
  ga[i] += go[i] * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x));
}`,
  // jeden wątek na wiersz: średnia i odwrotność odchylenia do st, wynik do o
  lnFwd: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gw: array<f32>;
@group(0) @binding(3) var<storage, read> bw: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read_write> st: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; } let d = p.m; let r = i * d;
  var mu = 0.0; for (var j = 0u; j < d; j++) { mu += x[r + j]; } mu /= f32(d);
  var v = 0.0; for (var j = 0u; j < d; j++) { let z = x[r + j] - mu; v += z * z; } v /= f32(d);
  let rs = inverseSqrt(v + 1e-5); st[2u * i] = mu; st[2u * i + 1u] = rs;
  for (var j = 0u; j < d; j++) { o[r + j] = (x[r + j] - mu) * rs * gw[j] + bw[j]; }
}`,
  lnBackRows: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gw: array<f32>;
@group(0) @binding(3) var<storage, read> st: array<f32>;
@group(0) @binding(4) var<storage, read> go: array<f32>;
@group(0) @binding(5) var<storage, read_write> gx: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; } let d = p.m; let r = i * d; let mu = st[2u * i]; let rs = st[2u * i + 1u];
  var s1 = 0.0; var s2 = 0.0;
  for (var j = 0u; j < d; j++) { let dxh = go[r + j] * gw[j]; s1 += dxh; s2 += dxh * (x[r + j] - mu) * rs; }
  for (var j = 0u; j < d; j++) { let xh = (x[r + j] - mu) * rs; gx[r + j] += rs * (go[r + j] * gw[j] - s1 / f32(d) - xh * s2 / f32(d)); }
}`,
  lnBackCols: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> st: array<f32>;
@group(0) @binding(3) var<storage, read> go: array<f32>;
@group(0) @binding(4) var<storage, read_write> gg: array<f32>;
@group(0) @binding(5) var<storage, read_write> gb: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x; if (j >= p.m) { return; } var a = 0.0; var b = 0.0;
  for (var i = 0u; i < p.n; i++) { let o = go[i * p.m + j]; a += o * (x[i * p.m + j] - st[2u * i]) * st[2u * i + 1u]; b += o; }
  gg[j] += a; gb[j] += b;
}`,
  // o[i,:] = E[idx[i],:] + P[i % T,:]   (embedding litery + embedding pozycji naraz; m = d, x = T)
  embFwd: `
@group(0) @binding(1) var<storage, read> E: array<f32>;
@group(0) @binding(2) var<storage, read> Pp: array<f32>;
@group(0) @binding(3) var<storage, read> idx: array<i32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let q = g.x; if (q >= p.n * p.m) { return; } let i = q / p.m; let j = q % p.m;
  o[q] = E[u32(idx[i]) * p.m + j] + Pp[(i % p.x) * p.m + j];
}`,
  // gradient embeddingów liter: wątek na (litera v, kolumna j) zbiera ze wszystkich wierszy z tą literą (bez wyścigów)
  embBackE: `
@group(0) @binding(1) var<storage, read> idx: array<i32>;
@group(0) @binding(2) var<storage, read> go: array<f32>;
@group(0) @binding(3) var<storage, read_write> gE: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let q = g.x; if (q >= p.k * p.m) { return; } let v = i32(q / p.m); let j = q % p.m; var s = 0.0;
  for (var i = 0u; i < p.n; i++) { if (idx[i] == v) { s += go[i * p.m + j]; } }
  gE[q] += s;
}`,
  embBackP: `
@group(0) @binding(1) var<storage, read> go: array<f32>;
@group(0) @binding(2) var<storage, read_write> gP: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let q = g.x; if (q >= p.x * p.m) { return; } let t = q / p.m; let j = q % p.m; var s = 0.0;
  for (var i = t; i < p.n; i += p.x) { s += go[i * p.m + j]; }
  gP[q] += s;
}`,
  // uwaga przyczynowa: wątek na (okno b, głowa h, pozycja i); p.n=B, p.k=T, p.m=d, p.x=głowy, f0=skala
  attFwd: `
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let T = p.k; let d = p.m; let nh = p.x; let hd = d / nh;
  let q = g.x; if (q >= p.n * nh * T) { return; }
  let i = q % T; let h = (q / T) % nh; let b = q / (T * nh);
  let ab = (b * nh + h) * T * T + i * T; let qi = (b * T + i) * 3u * d + h * hd;
  var mx = -1e30;
  for (var j = 0u; j <= i; j++) {
    let kj = (b * T + j) * 3u * d + d + h * hd; var s = 0.0;
    for (var k = 0u; k < hd; k++) { s += qkv[qi + k] * qkv[kj + k]; }
    s *= p.f0; A[ab + j] = s; mx = max(mx, s);
  }
  var sum = 0.0;
  for (var j = 0u; j <= i; j++) { let e = exp(A[ab + j] - mx); A[ab + j] = e; sum += e; }
  for (var j = i + 1u; j < T; j++) { A[ab + j] = 0.0; }
  let oi = (b * T + i) * d + h * hd;
  for (var k = 0u; k < hd; k++) { o[oi + k] = 0.0; }
  for (var j = 0u; j <= i; j++) {
    let pj = A[ab + j] / sum; A[ab + j] = pj; let vj = (b * T + j) * 3u * d + 2u * d + h * hd;
    for (var k = 0u; k < hd; k++) { o[oi + k] += pj * qkv[vj + k]; }
  }
}`,
  // wstecz, część 1: wątek na pozycję i liczy dS (do bufora S) i gradient zapytania Q_i
  attBackQ: `
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> go: array<f32>;
@group(0) @binding(4) var<storage, read_write> S: array<f32>;
@group(0) @binding(5) var<storage, read_write> gq: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let T = p.k; let d = p.m; let nh = p.x; let hd = d / nh;
  let q = g.x; if (q >= p.n * nh * T) { return; }
  let i = q % T; let h = (q / T) % nh; let b = q / (T * nh);
  let ab = (b * nh + h) * T * T + i * T; let oi = (b * T + i) * d + h * hd; let qi = (b * T + i) * 3u * d + h * hd;
  var dot = 0.0;
  for (var j = 0u; j <= i; j++) {
    let vj = (b * T + j) * 3u * d + 2u * d + h * hd; var s = 0.0;
    for (var k = 0u; k < hd; k++) { s += go[oi + k] * qkv[vj + k]; }
    S[ab + j] = s; dot += s * A[ab + j];
  }
  for (var j = 0u; j <= i; j++) {
    let ds = A[ab + j] * (S[ab + j] - dot) * p.f0; S[ab + j] = ds;
    let kj = (b * T + j) * 3u * d + d + h * hd;
    for (var k = 0u; k < hd; k++) { gq[qi + k] += ds * qkv[kj + k]; }
  }
}`,
  // wstecz, część 2: wątek na pozycję j zbiera gradienty klucza K_j i wartości V_j od wszystkich i ≥ j
  attBackKV: `
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> go: array<f32>;
@group(0) @binding(4) var<storage, read> S: array<f32>;
@group(0) @binding(5) var<storage, read_write> gq: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let T = p.k; let d = p.m; let nh = p.x; let hd = d / nh;
  let q = g.x; if (q >= p.n * nh * T) { return; }
  let j = q % T; let h = (q / T) % nh; let b = q / (T * nh);
  let kj = (b * T + j) * 3u * d + d + h * hd; let vj = kj + d;
  for (var i = j; i < T; i++) {
    let a = (b * nh + h) * T * T + i * T + j; let ds = S[a]; let pij = A[a];
    let qi = (b * T + i) * 3u * d + h * hd; let oi = (b * T + i) * d + h * hd;
    for (var k = 0u; k < hd; k++) { gq[kj + k] += ds * qkv[qi + k]; gq[vj + k] += pij * go[oi + k]; }
  }
}`,
  // softmax + entropia krzyżowa: błąd każdego wiersza i od razu gradient logitów (ten krok jest ostatni)
  ce: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<i32>;
@group(0) @binding(3) var<storage, read_write> gx: array<f32>;
@group(0) @binding(4) var<storage, read_write> loss: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; } let V = p.m; let r = i * V;
  var mx = -1e30; for (var j = 0u; j < V; j++) { mx = max(mx, x[r + j]); }
  var sum = 0.0; for (var j = 0u; j < V; j++) { sum += exp(x[r + j] - mx); }
  let t = u32(y[i]); loss[i] = log(sum) + mx - x[r + t];
  for (var j = 0u; j < V; j++) { var e = exp(x[r + j] - mx) / sum; if (j == t) { e -= 1.0; } gx[r + j] = e * p.f0; }
}`,
  // suma kwadratów gradientu (do przycięcia): jedna grupa 256 wątków na tensor, wynik dopisywany do acc[0]
  sumsq: `
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u) {
  var t = 0.0; for (var i = l.x; i < p.n; i += 256u) { t += s[i] * s[i]; } part[l.x] = t;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w /= 2u) { if (l.x < w) { part[l.x] += part[l.x + w]; } workgroupBarrier(); }
  if (l.x == 0u) { acc[0] += part[0]; }
}`,
  // Adam z przycięciem: skala = min(1, max / ‖g‖); st = [lr, 1−β1^t, 1−β2^t, max]
  adam: `
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read> gr: array<f32>;
@group(0) @binding(3) var<storage, read_write> m1: array<f32>;
@group(0) @binding(4) var<storage, read_write> m2: array<f32>;
@group(0) @binding(5) var<storage, read> acc: array<f32>;
@group(0) @binding(6) var<uniform> st: vec4f;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n) { return; }
  let nrm = sqrt(acc[0]); var sc = 1.0; if (nrm > st.w) { sc = st.w / nrm; }
  let gi = gr[i] * sc;
  m1[i] = 0.9 * m1[i] + 0.1 * gi; m2[i] = 0.999 * m2[i] + 0.001 * gi * gi;
  w[i] -= st.x * (m1[i] / st.y) / (sqrt(m2[i] / st.z) + 1e-8);
}`,
};

class GPUTensor {
  constructor(eng, size, shape, prev = [], back = null) { Object.assign(this, { eng, size, s: shape, prev, back }); this.buf = eng.buffer(size); this.g = null; }
  grad() { if (!this.g) { this.g = this.eng.buffer(this.size); this.eng.grads.push(this.g); } return this.g; }
}

class GPUEngine {
  static async create() {
    if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('Ta przeglądarka nie obsługuje WebGPU.');
    const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!ad) throw new Error('Brak karty graficznej dostępnej dla WebGPU.');
    const dev = await ad.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(8, ad.limits.maxStorageBuffersPerShaderStage) } });
    const info = ad.info || {};
    return new GPUEngine(dev, [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'WebGPU');
  }
  constructor(dev, name) { this.dev = dev; this.name = name; this.pipes = {}; this.grads = []; this.rec = null; this.lost = false; dev.lost.then(() => { this.lost = true; }); }
  buffer(n, data) {
    const b = this.dev.createBuffer({ size: Math.max(16, Math.ceil(n * 4 / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (data) this.dev.queue.writeBuffer(b, 0, data);
    return b;
  }
  pipe(name) {
    if (!this.pipes[name]) this.pipes[name] = this.dev.createComputePipeline({ layout: 'auto', compute: { module: this.dev.createShaderModule({ code: HDR + KERNELS[name] }), entryPoint: 'main' } });
    return this.pipes[name];
  }
  // Nagrywa jedno zadanie: program, bufory (od wiązania 1), parametry i liczbę grup wątków.
  run(name, bufs, prm, groups, extra) {
    const u = this.dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), ab = new ArrayBuffer(32), dv = new DataView(ab);
    ['n', 'k', 'm', 'x', 'y', 'z'].forEach((k, i) => dv.setUint32(i * 4, prm[k] || 0, true));
    dv.setFloat32(24, prm.f0 || 0, true); dv.setFloat32(28, prm.f1 || 0, true);
    this.dev.queue.writeBuffer(u, 0, ab);
    const pl = this.pipe(name), entries = [{ binding: 0, resource: { buffer: u } }, ...bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))];
    if (extra) entries.push({ binding: bufs.length + 1, resource: { buffer: extra } });
    this.rec.push({ pl, bg: this.dev.createBindGroup({ layout: pl.getBindGroupLayout(0), entries }), groups: Array.isArray(groups) ? groups : [groups] });
  }
  record(fn) { const prev = this.rec; this.rec = []; fn(); const r = this.rec; this.rec = prev; return r; }
  encode(enc, list) { const ps = enc.beginComputePass(); for (const d of list) { ps.setPipeline(d.pl); ps.setBindGroup(0, d.bg); ps.dispatchWorkgroups(...d.groups); } ps.end(); }
  async read(buf, n) {
    const st = this.dev.createBuffer({ size: Math.ceil(n * 4 / 16) * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, st, 0, Math.ceil(n * 4 / 16) * 16); this.dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ); const out = new Float32Array(st.getMappedRange().slice(0, n * 4)); st.unmap(); st.destroy(); return out;
  }
}
const grp = n => Math.ceil(n / WG);

// ---------- działania z gradientem (nagrywane, nie liczone od razu) ----------
function gMatmul(a, b) {
  const E = a.eng, [n, k] = a.s, m = b.s[1], o = new GPUTensor(E, n * m, [n, m], [a, b]);
  E.run('matmul', [a.buf, b.buf, o.buf], { n, k, m }, [Math.ceil(m / 16), Math.ceil(n / 16)]);
  o.back = () => {
    E.run('matmul', [o.g, b.buf, a.grad()], { n, k: m, m: k, y: 1, z: 1 }, [Math.ceil(k / 16), Math.ceil(n / 16)]);   // dA += dC·Bᵀ
    E.run('matmul', [a.buf, o.g, b.grad()], { n: k, k: n, m, x: 1, z: 1 }, [Math.ceil(m / 16), Math.ceil(k / 16)]);   // dB += Aᵀ·dC
  };
  return o;
}
function gAddBias(a, b) {
  const E = a.eng, [n, m] = a.s, o = new GPUTensor(E, n * m, a.s, [a, b]);
  E.run('addBias', [a.buf, b.buf, o.buf], { n: n * m, m }, grp(n * m));
  o.back = () => { E.run('axpy', [o.g, a.grad()], { n: n * m }, grp(n * m)); E.run('colsum', [o.g, b.grad()], { n, m }, grp(m)); };
  return o;
}
function gAdd(a, b) {
  const E = a.eng, o = new GPUTensor(E, a.size, a.s, [a, b]);
  E.run('add', [a.buf, b.buf, o.buf], { n: a.size }, grp(a.size));
  o.back = () => { E.run('axpy', [o.g, a.grad()], { n: a.size }, grp(a.size)); E.run('axpy', [o.g, b.grad()], { n: a.size }, grp(a.size)); };
  return o;
}
function gGelu(a) {
  const E = a.eng, o = new GPUTensor(E, a.size, a.s, [a]);
  E.run('gelu', [a.buf, o.buf], { n: a.size }, grp(a.size));
  o.back = () => E.run('geluBack', [a.buf, o.g, a.grad()], { n: a.size }, grp(a.size));
  return o;
}
function gLayerNorm(x, g, b) {
  const E = x.eng, [n, d] = x.s, o = new GPUTensor(E, n * d, x.s, [x, g, b]), st = E.buffer(2 * n);
  E.run('lnFwd', [x.buf, g.buf, b.buf, o.buf, st], { n, m: d }, grp(n));
  o.back = () => { E.run('lnBackRows', [x.buf, g.buf, st, o.g, x.grad()], { n, m: d }, grp(n)); E.run('lnBackCols', [x.buf, st, o.g, g.grad(), b.grad()], { n, m: d }, grp(d)); };
  return o;
}
function gEmbed(Emb, P, idx, n, T) {
  const E = Emb.eng, d = Emb.s[1], V = Emb.s[0], o = new GPUTensor(E, n * d, [n, d], [Emb, P]);
  E.run('embFwd', [Emb.buf, P.buf, idx, o.buf], { n, m: d, x: T }, grp(n * d));
  o.back = () => { E.run('embBackE', [idx, o.g, Emb.grad()], { n, k: V, m: d }, grp(V * d)); E.run('embBackP', [o.g, P.grad()], { n, m: d, x: T }, grp(T * d)); };
  return o;
}
function gAttention(qkv, B, T, nh) {
  const E = qkv.eng, d = qkv.s[1] / 3, o = new GPUTensor(E, B * T * d, [B * T, d], [qkv]), A = E.buffer(B * nh * T * T), S = E.buffer(B * nh * T * T);
  const prm = { n: B, k: T, m: d, x: nh, f0: 1 / Math.sqrt(d / nh) };
  E.run('attFwd', [qkv.buf, o.buf, A], prm, grp(B * nh * T));
  o.back = () => { E.run('attBackQ', [qkv.buf, A, o.g, S, qkv.grad()], prm, grp(B * nh * T)); E.run('attBackKV', [qkv.buf, A, o.g, S, qkv.grad()], prm, grp(B * nh * T)); };
  o.att = A;
  return o;
}
function gBackward(root) {
  const order = [], seen = new Set();
  (function visit(t) { if (seen.has(t)) return; seen.add(t); t.prev.forEach(visit); order.push(t); })(root);
  order.forEach(t => t.grad());
  for (let i = order.length - 1; i >= 0; i--) if (order[i].back) order[i].back();
}

// ---------- transformer na karcie ----------
// Wagi startowe bierze z modelu CPU (ten sam początek), na koniec oddaje je z powrotem.
class GPUTransformer {
  constructor(eng, m, B, S) {
    const { layers: L, dim: d, heads: nh, ctx: T } = m.hp, V = m.chars.length, dev = eng.dev;
    Object.assign(this, { eng, m, B, S, T, V });
    const P = m.params.map(p => { const t = new GPUTensor(eng, p.size, p.s); dev.queue.writeBuffer(t.buf, 0, p.d); return t; });
    this.P = P;
    this.m1 = P.map(p => eng.buffer(p.size)); this.m2 = P.map(p => eng.buffer(p.size)); this.snapB = null;
    this.acc = eng.buffer(4); this.stU = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const net = (idx, Bn) => {
      let k = 0; const Emb = P[k++], Pos = P[k++];
      let x = gEmbed(Emb, Pos, idx, Bn * T, T); const atts = [];
      for (let l = 0; l < L; l++) {
        const [g1, b1, Wqkv, bqkv, Wp, bp, g2, b2, W1, c1, W2, c2] = P.slice(k, k + 12); k += 12;
        const at = gAttention(gAddBias(gMatmul(gLayerNorm(x, g1, b1), Wqkv), bqkv), Bn, T, nh); atts.push(at);
        x = gAdd(x, gAddBias(gMatmul(at, Wp), bp));
        x = gAdd(x, gAddBias(gMatmul(gGelu(gAddBias(gMatmul(gLayerNorm(x, g2, b2), W1), c1)), W2), c2));
      }
      const [gf, bf, Wo, bo] = P.slice(k, k + 4);
      return { logits: gAddBias(gMatmul(gLayerNorm(x, gf, bf), Wo), bo), atts };
    };
    const lossOf = (logits, y, n) => { const loss = eng.buffer(n); eng.run('ce', [logits.buf, y, logits.grad(), loss], { n, m: V, f0: 1 / n }, grp(n)); return loss; };
    // plan nauki: w przód, błąd, wstecz, norma gradientu, Adam
    this.xT = eng.buffer(B * T); this.yT = eng.buffer(B * T);
    eng.grads = [];
    this.train = eng.record(() => {
      const { logits } = net(this.xT, B); this.lossT = lossOf(logits, this.yT, B * T);
      gBackward(logits);
      P.forEach(p => eng.run('sumsq', [p.g, this.acc], { n: p.size }, 1));
      P.forEach((p, i) => eng.run('adam', [p.buf, p.g, this.m1[i], this.m2[i], this.acc], { n: p.size }, grp(p.size), this.stU));
    });
    this.trainGrads = eng.grads.filter(g => g !== this.acc);
    // plan sprawdzania: tylko w przód, S okien naraz
    eng.grads = [];
    this.xE = eng.buffer(S * T); this.yE = eng.buffer(S * T);
    this.evalPlan = eng.record(() => { const { logits } = net(this.xE, S); this.lossE = lossOf(logits, this.yE, S * T); });
    this.t = 0;
  }
  // k kroków nauki naraz; batches: tablica {X, Y}
  steps(batches, lr, clip) {
    const { eng } = this, dev = eng.dev;
    for (const { X, Y } of batches) {
      this.t++;
      dev.queue.writeBuffer(this.xT, 0, X); dev.queue.writeBuffer(this.yT, 0, Y);
      dev.queue.writeBuffer(this.stU, 0, new Float32Array([lr, 1 - .9 ** this.t, 1 - .999 ** this.t, clip]));
      const enc = dev.createCommandEncoder();
      this.trainGrads.forEach(g => enc.clearBuffer(g)); enc.clearBuffer(this.acc);
      eng.encode(enc, this.train);
      dev.queue.submit([enc.finish()]);
    }
  }
  async lastLoss() { const r = await this.eng.read(this.lossT, this.B * this.T); let s = 0; for (const v of r) s += v; return s / r.length; }
  async evalRows(X, Y) {
    const { eng } = this, dev = eng.dev; dev.queue.writeBuffer(this.xE, 0, X); dev.queue.writeBuffer(this.yE, 0, Y);
    const enc = dev.createCommandEncoder(); eng.encode(enc, this.evalPlan); dev.queue.submit([enc.finish()]);
    return eng.read(this.lossE, this.S * this.T);
  }
  async done() { await this.eng.dev.queue.onSubmittedWorkDone(); }
  // wagi z karty do modelu CPU (próbki, mapy, zapis)
  async pull() { for (let i = 0; i < this.P.length; i++) this.m.params[i].d.set(await this.eng.read(this.P[i].buf, this.P[i].size)); }
  snapshot() {
    const dev = this.eng.dev; if (!this.snapB) this.snapB = this.P.map(p => this.eng.buffer(p.size));
    const enc = dev.createCommandEncoder(); this.P.forEach((p, i) => enc.copyBufferToBuffer(p.buf, 0, this.snapB[i], 0, this.snapB[i].size)); dev.queue.submit([enc.finish()]);
  }
  restore() {
    const dev = this.eng.dev, enc = dev.createCommandEncoder();
    this.P.forEach((p, i) => enc.copyBufferToBuffer(this.snapB[i], 0, p.buf, 0, this.snapB[i].size)); dev.queue.submit([enc.finish()]);
  }
}
