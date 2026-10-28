// SDF Walkers v0.16.2 · The specimen editor: one control per trait.
// ---------- specimen editor ----------
// Sides are the zombie's own: its right eye and right arm sit on your left in the Down view.
const names=a=>a.map(x=>x.name);
const EDIT=[
  {id:'e-sex',label:'Sex',opts:['male','female'],get:z=>z.sex,set:(z,v)=>z.sex=v},
  {id:'e-skin',label:'Skin',opts:names(SKINS),get:z=>SKINS.indexOf(z.skin),set:(z,v)=>z.skin=SKINS[v]},
  {id:'e-glow',label:'Eye glow',opts:names(EYES),get:z=>EYES.indexOf(z.eye),set:(z,v)=>z.eye=EYES[v]},
  {id:'e-eyes',label:'Eyes',opts:['both','right missing','left missing'],
    get:z=>z.eyeL===0?1:z.eyeR===0?2:0,
    set:(z,v)=>{z.eyeL=v===1?0:(z.eyeL||.065);z.eyeR=v===2?0:(z.eyeR||.06);}},
  {id:'e-skull',label:'Skull',opts:['intact','exposed'],get:z=>z.boneR?1:0,set:(z,v)=>{if(!v&&z.boneR)z.boneKeep=z.boneR;z.boneR=v?(z.boneKeep||.3):0;}},
  {id:'e-teeth',label:'Teeth',opts:['full set','gap 1','gap 2','gap 3','gap 4','gap 5'],get:z=>z.missing+1,set:(z,v)=>z.missing=v-1},
  {id:'e-wound',label:'Wound',opts:['left cheek','right cheek'],get:z=>z.wound>0?0:1,set:(z,v)=>z.wound=v===0?1:-1},
  {id:'e-arms',label:'Arms',opts:['both','right missing','left missing'],get:z=>z.armGone===0?0:z.armGone<0?1:2,set:(z,v)=>z.armGone=[0,-1,1][v]},
  {id:'e-shirt',label:'Shirt',opts:names(CLOTH),get:z=>CLOTH.indexOf(z.shirt),set:(z,v)=>z.shirt=CLOTH[v]},
  {id:'e-pants',label:'Trousers',opts:names(CLOTH),get:z=>CLOTH.indexOf(z.pants),set:(z,v)=>z.pants=CLOTH[v]},
  {id:'e-cloth',label:'Clothing',opts:names(GARMENTS),get:z=>z.cloth,set:(z,v)=>z.cloth=v},
  {id:'e-shoe',label:'Shoes',opts:names(SHOES),get:z=>z.shoe,set:(z,v)=>z.shoe=v},
  {id:'e-hair',label:'Hair',opts:names(HAIR_STYLES),get:z=>z.hair,set:(z,v)=>z.hair=v},
  {id:'e-hairc',label:'Hair colour',opts:names(HAIRS),get:z=>HAIRS.indexOf(z.hairC),set:(z,v)=>z.hairC=HAIRS[v]},
  {id:'e-head',label:'Head size',range:[.16,.3,.01],get:z=>z.head,set:(z,v)=>z.head=v,fmt:v=>Math.round(v/.2*100)+'%'},
  {id:'e-jaw',label:'Jaw',range:[.06,.2,.01],get:z=>z.jaw,set:(z,v)=>z.jaw=v,fmt:v=>v<.1?'clenched':v<.14?'ajar':'gaping'},
  {id:'e-tilt',label:'Head tilt',range:[-.3,.3,.02],get:z=>z.tilt,set:(z,v)=>z.tilt=v,fmt:v=>Math.round(v*57.3)+'°'},
  {id:'e-hunch',label:'Hunch',range:[.08,.5,.01],get:z=>z.hunch,set:(z,v)=>z.hunch=v,fmt:v=>v>.35?'heavy':v>.22?'marked':'slight'},
  {id:'e-height',label:'Height',range:[.88,1.12,.01],get:z=>z.height,set:(z,v)=>z.height=v,fmt:v=>Math.round(v*100)+'%'},
  {id:'e-build',label:'Build',range:[0,1,.05],get:z=>z.build,set:(z,v)=>z.build=v,fmt:v=>v<.33?'slim':v<.66?'average':'heavy'},
  {id:'e-shoulder',label:'Shoulders',range:[0,1,.05],get:z=>z.shoulder,set:(z,v)=>z.shoulder=v,fmt:v=>v<.33?'narrow':v<.66?'average':'broad'},
  {id:'e-limp',label:'Limp',range:[0,.85,.05],get:z=>z.limp,set:(z,v)=>z.limp=v,fmt:v=>v===0?'none':Math.round(v*100)+'%'},
];
const editor=document.getElementById('editor');
EDIT.forEach(f=>{
  const lab=document.createElement('label');lab.htmlFor=f.id;
  if(f.opts){
    lab.innerHTML=`<span>${f.label}</span>`;
    const sel=document.createElement('select');sel.id=f.id;
    f.opts.forEach((o,i)=>sel.add(new Option(o,i)));
    sel.addEventListener('change',()=>{f.set(Z,+sel.value);edited();});
    lab.appendChild(sel);f.el=sel;
  }else{
    lab.innerHTML=`<span>${f.label}<output id="${f.id}-o"></output></span>`;
    const r=document.createElement('input');r.type='range';r.id=f.id;
    [r.min,r.max,r.step]=f.range;
    r.addEventListener('input',()=>{f.set(Z,+r.value);f.out.textContent=f.fmt(+r.value);edited();});
    lab.appendChild(r);f.el=r;f.out=lab.querySelector('output');
  }
  editor.appendChild(lab);
});
function syncEditor(){
  EDIT.forEach(f=>{const v=f.get(Z);f.el.value=v;if(f.out)f.out.textContent=f.fmt(v);});
  document.getElementById('editedTag').hidden=true;
  document.getElementById('seedLabel').textContent='SPECIMEN #'+Z.seed;
  if(typeof drawHud==='function'&&window.hudReady)drawHud();
}
let debounce=0;
function edited(){
  document.getElementById('editedTag').hidden=false;
  document.getElementById('seedLabel').textContent='SPECIMEN #'+Z.seed+' · EDITED';
  clearTimeout(debounce);debounce=setTimeout(rebuild,140);
  drawHud();
}

