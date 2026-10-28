// SDF Walkers v0.17.0 · Palettes, trait vocabularies, animation list and the seeded specimen generator.
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
// ---------- palettes ----------
const SKINS=[
  {name:'rotten green',h:95,s:.25,c:[[38,34,56],[64,82,70],[104,128,86],[156,172,112]]},
  {name:'ash',h:220,s:.05,c:[[36,32,52],[78,80,92],[120,124,128],[170,172,164]]},
  {name:'drowned',h:205,s:.15,c:[[34,28,54],[70,74,104],[110,124,140],[160,178,172]]},
  {name:'ochre',h:45,s:.3,c:[[40,30,44],[92,78,60],[140,124,80],[188,176,116]]},
];
const EYES=[{name:'yellow',h:52,c:[250,238,150]},{name:'red',h:8,c:[240,96,72]},{name:'milky',h:200,c:[205,228,236]}];
const HAIR_STYLES=[{name:'bald'},{name:'short'},{name:'long'},{name:'patchy'}];
const HAIRS=[{name:'black',c:[[18,14,22],[32,26,34],[50,42,50],[70,60,68]]},{name:'grey',c:[[46,44,54],[90,88,96],[134,132,138],[176,174,178]]},
  {name:'brown',c:[[32,20,18],[60,38,28],[92,60,40],[122,84,56]]},{name:'dirty blonde',c:[[48,38,28],[100,84,54],[146,126,80],[184,166,110]]}];
const SHOES=[{name:'barefoot',c:[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]},
  {name:'sneakers',c:[[60,58,70],[120,120,128],[180,180,184],[224,222,218]]},
  {name:'work boots',c:[[34,22,18],[66,44,30],[98,66,42],[130,92,60]]},
  {name:'dress shoes',c:[[12,10,14],[24,20,26],[40,34,42],[62,54,64]]},
  {name:'rubber boots',c:[[20,34,26],[34,62,42],[52,92,60],[78,122,82]]}];
const GARMENTS=[{name:'shirt'},{name:'jacket'},{name:'hospital gown'},{name:'sleeveless'},{name:'overalls'},{name:'rags'},{name:'shirtless / crop top'},{name:'dress'},{name:'shorts'}];
const CLOTH=[{name:'dirty violet',h:265},{name:'faded blue',h:215,l:.05},{name:'hospital green',h:160,l:.03},{name:'rust',h:15,l:-.02},{name:'mustard',h:45,l:.04},{name:'grey denim',h:230,s:.08,l:.1}];
function hsl(h,s,l){h=((h%360)+360)%360/360;const f=n=>{const k=(n+h*12)%12,a=s*Math.min(l,1-l);return l-a*Math.max(-1,Math.min(k-3,9-k,1));};return [f(0),f(8),f(4)].map(v=>Math.round(v*255));}
const ramp=(h,s=.24,dl=0)=>[hsl(h+18,s,.12+dl),hsl(h+8,s,.24+dl),hsl(h,s*.92,.36+dl),hsl(h-6,s*.85,.48+dl)];
const hueDist=(a,b)=>{const d=Math.abs(a-b)%360;return d>180?360-d:d;};
const FIXED={2:[[24,14,24],[34,18,30],[52,26,40],[70,40,52]],
  3:[[120,104,80],[170,155,110],[214,200,150],[230,220,175]],
  4:[[80,70,78],[130,120,112],[176,164,140],[214,204,176]],
  5:[[50,12,24],[96,24,36],[140,40,44],[170,60,56]]};

// ---------- animations ----------
const ANIMS=[
  {key:'walk',label:'Walk',fps:9,loop:true},
  {key:'idle',label:'Idle',fps:6,loop:true},
  {key:'attack',label:'Attack',fps:12,loop:true},
  {key:'hit',label:'Hit',fps:14,loop:false},
  {key:'death',label:'Death',fps:10,loop:false},
];
const DIRS=[{key:'down',yaw:0},{key:'left',yaw:Math.PI/2},{key:'right',yaw:-Math.PI/2},{key:'up',yaw:Math.PI}];
const FRAMES=8;

// ---------- specimens ----------
const rulesEl=document.getElementById('rules');
function mulberry(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
// Oklab: a colour space where distance matches how different two colours look
function oklab([r,g,b]){
  const f=c=>{c/=255;return c<=.04045?c/12.92:Math.pow((c+.055)/1.055,2.4);};
  r=f(r);g=f(g);b=f(b);
  const l=Math.cbrt(.4122214708*r+.5363325363*g+.0514459929*b),m=Math.cbrt(.2119034982*r+.6806995451*g+.1073969566*b),s=Math.cbrt(.0883024619*r+.2817188376*g+.6299787005*b);
  return[.2104542553*l+.793617785*m-.0040720468*s,1.9779984951*l-2.428592205*m+.4505937099*s,.0259040371*l+.7827717662*m-.808675766*s];
}
const labDist=(a,b)=>{const p=oklab(a),q=oklab(b);return Math.hypot(p[0]-q[0],p[1]-q[1],p[2]-q[2]);};
// the lit mid-tone of each material, which is what the eye compares
const midShirt=c=>ramp(c.h,c.s??.24,c.l??0)[2],midPants=c=>ramp(c.h,(c.s??.24)*.7,(c.l??0)-.04)[2],midSkin=s=>s.c[2];
function contrasts(skin,shirt,pants){return{sp:labDist(midShirt(shirt),midPants(pants)),ss:labDist(midShirt(shirt),midSkin(skin)),ps:labDist(midPants(pants),midSkin(skin))};}
const MIN_CONTRAST=.07;
function paletteOk(z){
  const sk=z.skin,k=contrasts(z.skin,z.shirt,z.pants);
  if(k.sp<MIN_CONTRAST||k.ss<MIN_CONTRAST)return false;                  // also catches grey on grey and dark on dark
  if(hueDist(z.shirt.h,z.pants.h)<50)return false;                       // shirt and trousers must read as two things
  if(sk.s>.1&&hueDist(z.shirt.h,sk.h)<35)return false;                   // shirt must not melt into the skin
  if(hueDist(z.eye.h,sk.h)<30)return false;                              // eye glow must stand out against the skin
  if(sk.name==='ash'&&z.eye.name==='milky')return false;
  return true;
}
function makeZombie(sd){
  const r=mulberry(sd),pick=a=>a[Math.floor(r()*a.length)];
  const z={seed:sd,skin:pick(SKINS),eye:pick(EYES),shirt:pick(CLOTH),pants:pick(CLOTH),
    tilt:(r()-.5)*.45,jaw:.08+r()*.1,
    eyeL:r()<.15?0:.05+r()*.03,eyeR:r()<.15?0:.045+r()*.03,
    boneR:r()<.3?0:.22+r()*.15,boneX:(r()-.5)*.6,
    missing:r()<.7?Math.floor(r()*5):-1,wound:r()<.5?1:-1,
    hunch:.12+r()*.35,armGone:r()<.3?(r()<.5?-1:1):0,limp:r()<.35?.55+r()*.3:0};
  if(z.eyeL===0&&z.eyeR===0)z.eyeR=.06;
  z.head=.21+r()*.05;
  z.height=.9+r()*.2;z.build=r();z.shoulder=r();
  const hr=r();z.hair=hr<.25?0:hr<.55?1:hr<.8?2:3;
  z.hairC=pick(HAIRS);z.sex=r()<.5?1:0;
  if(z.sex&&z.hair===0&&r()<.75)z.hair=2;                 // most women zombies keep their hair
  const garb=z.sex?[0,1,2,3,4,5,6,7,7,8]:[0,0,1,2,3,4,5,6,8];
  z.cloth=garb[Math.floor(r()*garb.length)];
  if(z.sex)z.height=.88+r()*.16;
  const sh=r();z.shoe=sh<.15?0:sh<.4?1:sh<.65?2:sh<.85?3:4;
  if(rulesEl.checked){
    for(let k=0;k<40&&!paletteOk(z);k++){z.shirt=pick(CLOTH);z.pants=pick(CLOTH);z.eye=pick(EYES);}
  }
  return z;
}
// compact, storable form of a specimen (indices instead of palette objects)
function ser(z){return{seed:z.seed,skin:SKINS.indexOf(z.skin),eye:EYES.indexOf(z.eye),shirt:CLOTH.indexOf(z.shirt),pants:CLOTH.indexOf(z.pants),
  head:z.head,sex:z.sex,height:z.height,build:z.build,shoulder:z.shoulder,hair:z.hair,hairC:HAIRS.indexOf(z.hairC),cloth:z.cloth,shoe:z.shoe,jaw:z.jaw,tilt:z.tilt,hunch:z.hunch,limp:z.limp,boneR:z.boneR,boneX:z.boneX,armGone:z.armGone,eyeL:z.eyeL,eyeR:z.eyeR,missing:z.missing,wound:z.wound};}
function des(t){return{height:1,build:.5,shoulder:.5,hair:0,cloth:0,sex:0,shoe:3,...t,skin:SKINS[t.skin],eye:EYES[t.eye],shirt:CLOTH[t.shirt],pants:CLOTH[t.pants],hairC:HAIRS[t.hairC??0]};}
function paletteArray(z){
  const pal=[];
  for(let m=0;m<10;m++){
    const set=m===9?SHOES[z.shoe].c:m===8?z.hairC.c:m===0?z.skin.c:m===1?Array(4).fill(z.eye.c):m===6?ramp(z.shirt.h,z.shirt.s??.24,z.shirt.l??0):m===7?ramp(z.pants.h,(z.pants.s??.24)*.7,(z.pants.l??0)-.04):FIXED[m];
    set.forEach(c=>pal.push(c[0]/255,c[1]/255,c[2]/255));
  }
  return new Float32Array(pal);
}

