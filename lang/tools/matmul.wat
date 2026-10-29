;; Laboratorium języka · mnożenie macierzy w WebAssembly z SIMD (f32x4: 4 liczby jedną instrukcją).
;; Źródło dla lang/js/wasm.js; kompilacja: node lang/tools/build_wasm.js
;; Wszystkie macierze float32, wiersz po wierszu; adresy w bajtach. acc=0: wynik nadpisuje C, acc=1: dodaje.
(module
  (memory (export "mem") 1)

  ;; C[n,m] (+)= A[n,k] · B[k,m]      (kolejność i-p-j: wiersz B mnożony przez jedną liczbę A, 4 kolumny naraz)
  (func (export "mm") (param $a i32) (param $b i32) (param $c i32) (param $n i32) (param $k i32) (param $m i32) (param $acc i32)
    (local $i i32) (local $p i32) (local $j i32) (local $crow i32) (local $brow i32) (local $av f32) (local $vs v128) (local $m4 i32) (local $q i32)
    (if (i32.eqz (local.get $acc)) (then (memory.fill (local.get $c) (i32.const 0) (i32.mul (i32.mul (local.get $n) (local.get $m)) (i32.const 4)))))
    (local.set $m4 (i32.and (local.get $m) (i32.const -4)))
    (block $bi (loop $li
      (br_if $bi (i32.ge_u (local.get $i) (local.get $n)))
      (local.set $crow (i32.add (local.get $c) (i32.shl (i32.mul (local.get $i) (local.get $m)) (i32.const 2))))
      (local.set $p (i32.const 0))
      (block $bp (loop $lp
        (br_if $bp (i32.ge_u (local.get $p) (local.get $k)))
        (local.set $av (f32.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $k)) (local.get $p)) (i32.const 2)))))
        (if (f32.ne (local.get $av) (f32.const 0)) (then
          (local.set $vs (f32x4.splat (local.get $av)))
          (local.set $brow (i32.add (local.get $b) (i32.shl (i32.mul (local.get $p) (local.get $m)) (i32.const 2))))
          (local.set $j (i32.const 0))
          (block $bj (loop $lj
            (br_if $bj (i32.ge_u (local.get $j) (local.get $m4)))
            (local.set $q (i32.shl (local.get $j) (i32.const 2)))
            (v128.store (i32.add (local.get $crow) (local.get $q))
              (f32x4.add (v128.load (i32.add (local.get $crow) (local.get $q)))
                         (f32x4.mul (local.get $vs) (v128.load (i32.add (local.get $brow) (local.get $q))))))
            (local.set $j (i32.add (local.get $j) (i32.const 4)))
            (br $lj)))
          (block $bt (loop $lt
            (br_if $bt (i32.ge_u (local.get $j) (local.get $m)))
            (local.set $q (i32.shl (local.get $j) (i32.const 2)))
            (f32.store (i32.add (local.get $crow) (local.get $q))
              (f32.add (f32.load (i32.add (local.get $crow) (local.get $q)))
                       (f32.mul (local.get $av) (f32.load (i32.add (local.get $brow) (local.get $q))))))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $lt)))))
        (local.set $p (i32.add (local.get $p) (i32.const 1)))
        (br $lp)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $li))))

  ;; C[n,m] (+)= A[n,k] · Bᵀ, gdzie B zapisane jako [m,k]: C[i,j] = iloczyn skalarny wiersza i z A i wiersza j z B
  (func (export "mmTB") (param $a i32) (param $b i32) (param $c i32) (param $n i32) (param $k i32) (param $m i32) (param $acc i32)
    (local $i i32) (local $j i32) (local $p i32) (local $arow i32) (local $brow i32) (local $s v128) (local $t f32) (local $k4 i32) (local $q i32) (local $ca i32)
    (local.set $k4 (i32.and (local.get $k) (i32.const -4)))
    (block $bi (loop $li
      (br_if $bi (i32.ge_u (local.get $i) (local.get $n)))
      (local.set $arow (i32.add (local.get $a) (i32.shl (i32.mul (local.get $i) (local.get $k)) (i32.const 2))))
      (local.set $j (i32.const 0))
      (block $bj (loop $lj
        (br_if $bj (i32.ge_u (local.get $j) (local.get $m)))
        (local.set $brow (i32.add (local.get $b) (i32.shl (i32.mul (local.get $j) (local.get $k)) (i32.const 2))))
        (local.set $s (v128.const f32x4 0 0 0 0))
        (local.set $p (i32.const 0))
        (block $bp (loop $lp
          (br_if $bp (i32.ge_u (local.get $p) (local.get $k4)))
          (local.set $q (i32.shl (local.get $p) (i32.const 2)))
          (local.set $s (f32x4.add (local.get $s) (f32x4.mul (v128.load (i32.add (local.get $arow) (local.get $q))) (v128.load (i32.add (local.get $brow) (local.get $q))))))
          (local.set $p (i32.add (local.get $p) (i32.const 4)))
          (br $lp)))
        (local.set $t (f32.add (f32.add (f32x4.extract_lane 0 (local.get $s)) (f32x4.extract_lane 1 (local.get $s)))
                               (f32.add (f32x4.extract_lane 2 (local.get $s)) (f32x4.extract_lane 3 (local.get $s)))))
        (block $bt (loop $lt
          (br_if $bt (i32.ge_u (local.get $p) (local.get $k)))
          (local.set $q (i32.shl (local.get $p) (i32.const 2)))
          (local.set $t (f32.add (local.get $t) (f32.mul (f32.load (i32.add (local.get $arow) (local.get $q))) (f32.load (i32.add (local.get $brow) (local.get $q))))))
          (local.set $p (i32.add (local.get $p) (i32.const 1)))
          (br $lt)))
        (local.set $ca (i32.add (local.get $c) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $m)) (local.get $j)) (i32.const 2))))
        (if (local.get $acc) (then (local.set $t (f32.add (local.get $t) (f32.load (local.get $ca))))))
        (f32.store (local.get $ca) (local.get $t))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $lj)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $li))))

  ;; C[n,m] (+)= Aᵀ · B, gdzie A zapisane jako [k,n]: dla każdego p wiersz p z B dodawany do wierszy C z wagami A[p,i]
  (func (export "mmTA") (param $a i32) (param $b i32) (param $c i32) (param $n i32) (param $k i32) (param $m i32) (param $acc i32)
    (local $i i32) (local $p i32) (local $j i32) (local $crow i32) (local $brow i32) (local $av f32) (local $vs v128) (local $m4 i32) (local $q i32)
    (if (i32.eqz (local.get $acc)) (then (memory.fill (local.get $c) (i32.const 0) (i32.mul (i32.mul (local.get $n) (local.get $m)) (i32.const 4)))))
    (local.set $m4 (i32.and (local.get $m) (i32.const -4)))
    (block $bp (loop $lp
      (br_if $bp (i32.ge_u (local.get $p) (local.get $k)))
      (local.set $brow (i32.add (local.get $b) (i32.shl (i32.mul (local.get $p) (local.get $m)) (i32.const 2))))
      (local.set $i (i32.const 0))
      (block $bi (loop $li
        (br_if $bi (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $av (f32.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $p) (local.get $n)) (local.get $i)) (i32.const 2)))))
        (if (f32.ne (local.get $av) (f32.const 0)) (then
          (local.set $vs (f32x4.splat (local.get $av)))
          (local.set $crow (i32.add (local.get $c) (i32.shl (i32.mul (local.get $i) (local.get $m)) (i32.const 2))))
          (local.set $j (i32.const 0))
          (block $bj (loop $lj
            (br_if $bj (i32.ge_u (local.get $j) (local.get $m4)))
            (local.set $q (i32.shl (local.get $j) (i32.const 2)))
            (v128.store (i32.add (local.get $crow) (local.get $q))
              (f32x4.add (v128.load (i32.add (local.get $crow) (local.get $q)))
                         (f32x4.mul (local.get $vs) (v128.load (i32.add (local.get $brow) (local.get $q))))))
            (local.set $j (i32.add (local.get $j) (i32.const 4)))
            (br $lj)))
          (block $bt (loop $lt
            (br_if $bt (i32.ge_u (local.get $j) (local.get $m)))
            (local.set $q (i32.shl (local.get $j) (i32.const 2)))
            (f32.store (i32.add (local.get $crow) (local.get $q))
              (f32.add (f32.load (i32.add (local.get $crow) (local.get $q)))
                       (f32.mul (local.get $av) (f32.load (i32.add (local.get $brow) (local.get $q))))))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $lt)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $li)))
      (local.set $p (i32.add (local.get $p) (i32.const 1)))
      (br $lp))))
)
