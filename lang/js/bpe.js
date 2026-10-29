// Laboratorium języka · tokenizer BPE (byte-pair encoding, Sennrich i in. 2016; wariant z GPT-2).
//
// Start: każda litera to osobny token. Potem w kółko: znajdź parę sąsiednich tokenów, która w tekście
// występuje najczęściej, i sklej ją w nowy token („r”+„z” → „rz”, potem „prz”, „przy”…). Po N sklejeniach
// częste słowa i końcówki są pojedynczymi tokenami, a rzadkie słowa składają się z kilku kawałków.
// Tekst najpierw dzielimy na słowa (spacja należy do słowa po niej), więc token nigdy nie przechodzi przez
// granicę słowa. Liczymy pary na rodzajach słów razem z ich liczbą wystąpień, a nie na całym tekście,
// dzięki czemu nauka trwa sekundy.

const BPE_SPLIT = / ?[\p{L}\p{M}\p{N}]+| ?[^\s\p{L}\p{M}\p{N}]+|\n|[^\S\n]+/gu;

function bpeTrain(text, nMerges, onProgress) {
  const base = [...new Set(text)].sort((a, b) => a.localeCompare(b, 'pl'));
  const id = new Map(base.map((c, i) => [c, i])), vocab = base.slice(), merges = [];
  // rodzaje słów i ich liczność
  const wc = new Map();
  for (const w of text.match(BPE_SPLIT) || []) wc.set(w, (wc.get(w) || 0) + 1);
  const seqs = [], cnt = [];
  for (const [w, c] of wc) { seqs.push([...w].map(ch => id.get(ch))); cnt.push(c); }
  // liczność par i indeks: para → słowa, w których występuje
  const K = 1 << 20, key = (a, b) => a * K + b;
  const pc = new Map(), pw = new Map();
  const addPairs = (wi, sgn) => {
    const s = seqs[wi];
    for (let i = 0; i + 1 < s.length; i++) {
      const k = key(s[i], s[i + 1]);
      pc.set(k, (pc.get(k) || 0) + sgn * cnt[wi]);
      if (sgn > 0) { let set = pw.get(k); if (!set) pw.set(k, set = new Set()); set.add(wi); }
    }
  };
  for (let wi = 0; wi < seqs.length; wi++) addPairs(wi, 1);
  for (let r = 0; r < nMerges; r++) {
    let best = -1, bc = 1;
    for (const [k, c] of pc) if (c > bc || (c === bc && best >= 0 && k < best)) { best = k; bc = c; }
    if (best < 0) break;                               // żadna para nie występuje dwa razy: koniec
    const a = Math.floor(best / K), b = best % K, nid = vocab.length;
    vocab.push(vocab[a] + vocab[b]); merges.push([a, b]);
    for (const wi of pw.get(best) || []) {
      const s = seqs[wi]; let has = false;
      for (let i = 0; i + 1 < s.length; i++) if (s[i] === a && s[i + 1] === b) { has = true; break; }
      if (!has) continue;
      addPairs(wi, -1);
      const out = [];
      for (let i = 0; i < s.length; i++) { if (i + 1 < s.length && s[i] === a && s[i + 1] === b) { out.push(nid); i++; } else out.push(s[i]); }
      seqs[wi] = out;
      addPairs(wi, 1);
    }
    pc.delete(best); pw.delete(best);
    if (onProgress && r % 32 === 31) onProgress(r + 1, nMerges);
  }
  return bpeTokenizer(vocab, merges, base.length);
}

// Tokenizer z gotowego słownika i listy sklejeń (też po wczytaniu modelu z pliku).
function bpeTokenizer(vocab, merges, nBase) {
  const K = 1 << 20, rank = new Map(merges.map(([a, b], r) => [a * K + b, r]));
  const id = new Map(vocab.slice(0, nBase).map((c, i) => [c, i])), cache = new Map();
  const word = w => {
    let c = cache.get(w); if (c) return c;
    let s = [...w].map(ch => id.get(ch)).filter(v => v !== undefined);   // nieznane znaki pomijamy
    while (s.length > 1) {                              // sklejaj zawsze parę o najniższym numerze sklejenia, jak przy nauce
      let br = Infinity, bi = -1;
      for (let i = 0; i + 1 < s.length; i++) { const r = rank.get(s[i] * K + s[i + 1]); if (r !== undefined && r < br) { br = r; bi = i; } }
      if (bi < 0) break;
      s = [...s.slice(0, bi), nBase + br, ...s.slice(bi + 2)];
    }
    if (cache.size < 200000) cache.set(w, s);
    return s;
  };
  return {
    vocab, merges, nBase,
    encode(text) { const out = []; for (const w of text.match(BPE_SPLIT) || []) for (const t of word(w)) out.push(t); return out; },
    decode(ids) { return ids.map(i => vocab[i]).join(''); },
  };
}
// Litery jako „tokenizer” bez sklejeń: ten sam interfejs dla obu jednostek.
function charTokenizer(chars) {
  const id = new Map(chars.map((c, i) => [c, i]));
  return { vocab: chars, merges: [], nBase: chars.length,
    encode(text) { const out = []; for (const ch of text) { const v = id.get(ch); if (v !== undefined) out.push(v); } return out; },
    decode(ids) { return ids.map(i => chars[i]).join(''); } };
}
if (typeof module !== 'undefined') module.exports = { bpeTrain, bpeTokenizer, charTokenizer };
