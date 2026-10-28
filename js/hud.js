// SDF Walkers v0.17.0 · The movable see-through network view.
// ---------- network HUD ----------
const hud=document.getElementById('hud'),hudCv=document.getElementById('hudCv'),hctx=hudCv.getContext('2d'),hudOnEl=document.getElementById('hudOn');
const HW=520,HH=745;
let hudFocus=null,think=null;
const GROUPS=[['Skin',4],['Eye glow',3],['Shirt',6],['Trousers',6],['Hair',4],['Hair colour',4],['Clothing',9],['Shoes',5],['Body',12],['Colour gaps',3]];
function hudOpen(){return !hud.hidden;}
function setHud(on){
  hud.hidden=!on;hudOnEl.checked=on;
  try{localStorage.setItem('sdf-walkers-hud',on?'1':'0');}catch(e){}
  if(on){sizeHud();applyHudPos();drawHud();}
}
// position and width the viewer chose; null = default corner
let hudPos=null;
try{hudPos=JSON.parse(localStorage.getItem('sdf-walkers-hudpos')||'null');}catch(e){hudPos=null;}
function applyHudPos(){
  if(!hudPos){hud.style.cssText='';return;}
  const maxW=Math.min(innerWidth-16,(innerHeight-90)*HW/HH);
  const w=Math.max(Math.min(hudPos.w,maxW),Math.min(300,innerWidth-16));
  hud.style.width=w+'px';
  const h=hud.offsetHeight||w*HH/HW;
  const x=Math.min(Math.max(hudPos.x,0),Math.max(0,innerWidth-w));
  const y=Math.min(Math.max(hudPos.y,0),Math.max(0,innerHeight-h));
  Object.assign(hud.style,{left:x+'px',top:y+'px',right:'auto',bottom:'auto'});
}
function saveHudPos(){try{localStorage.setItem('sdf-walkers-hudpos',JSON.stringify(hudPos));}catch(e){}}
function dragWith(el,onMove){
  el.addEventListener('pointerdown',e=>{
    if(e.target.closest('button'))return;
    e.preventDefault();
    const r=hud.getBoundingClientRect(),sx=e.clientX,sy=e.clientY;
    el.setPointerCapture(e.pointerId);
    const move=ev=>{onMove(r,ev.clientX-sx,ev.clientY-sy);applyHudPos();};
    const up=()=>{el.removeEventListener('pointermove',move);el.removeEventListener('pointerup',up);el.removeEventListener('pointercancel',up);saveHudPos();};
    el.addEventListener('pointermove',move);el.addEventListener('pointerup',up);el.addEventListener('pointercancel',up);
  });
}
function initHudDrag(){
  const head=document.getElementById('hudHead'),grip=document.getElementById('hudGrip');
  dragWith(head,(r,dx,dy)=>{hudPos={x:r.left+dx,y:r.top+dy,w:r.width};});
  dragWith(grip,(r,dx)=>{hudPos={x:r.left,y:r.top,w:r.width+dx};});
  head.addEventListener('dblclick',e=>{if(e.target.closest('button'))return;hudPos=null;saveHudPos();applyHudPos();});
  head.addEventListener('keydown',e=>{
    const r=hud.getBoundingClientRect(),step=e.shiftKey?60:20;
    const k={ArrowLeft:[-step,0,0],ArrowRight:[step,0,0],ArrowUp:[0,-step,0],ArrowDown:[0,step,0],'+':[0,0,40],'=':[0,0,40],'-':[0,0,-40]}[e.key];
    if(!k)return;
    e.preventDefault();hudPos={x:r.left+k[0],y:r.top+k[1],w:r.width+k[2]};applyHudPos();saveHudPos();
  });
  addEventListener('resize',()=>{if(!hud.hidden)applyHudPos();});
}
function sizeHud(){const d=window.devicePixelRatio||1;hudCv.width=HW*d;hudCv.height=HH*d;hctx.setTransform(d,0,0,d,0,0);}
function drawHud(){
  if(hud.hidden)return;
  const c=hctx,m=training||net,t=hudFocus||ser(Z),x=feats(t);
  const ROT='156,172,112',BLD='208,96,94',FG='221,214,196',MU='143,136,158';
  const font=(px,bold)=>`${bold?'700 ':''}${px}px "Atkinson Hyperlegible",system-ui,sans-serif`;
  c.clearRect(0,0,HW,HH);
  const inX=120,hidX=290,outX=392,top=24,gap=7.6;
  // input column, grouped by trait
  const inY=[];let yy=top;
  c.textBaseline='middle';
  // each group gets a bracket; one-hot groups name the active option, number groups label every input
  const ACTIVE=[SKINS[t.skin].name,EYES[t.eye].name,CLOTH[t.shirt].name,CLOTH[t.pants].name,HAIR_STYLES[t.hair??0].name,HAIRS[t.hairC??0].name,GARMENTS[t.cloth??0].name,SHOES[t.shoe??3].name];
  const NUM={8:['height','build','shoulders','female','head size','jaw open','head tilt','hunch','limp','bare skull','arm gone','eye gone'],9:['shirt↔trousers','shirt↔skin','trousers↔skin']};
  GROUPS.forEach(([name,n],g)=>{
    if(NUM[g])yy+=11;   // room for the group heading above the labelled inputs
    const y0=yy;for(let k=0;k<n;k++){inY.push(yy);yy+=gap;}
    const y1=yy-gap;
    c.strokeStyle=`rgba(${MU},.45)`;c.lineWidth=1;c.beginPath();c.moveTo(inX-7,y0-2);c.lineTo(inX-9,y0-2);c.lineTo(inX-9,y1+2);c.lineTo(inX-7,y1+2);c.stroke();
    c.textAlign='right';
    if(NUM[g]){
      c.fillStyle=`rgb(${MU})`;c.font=font(8);
      NUM[g].forEach((lab,k)=>c.fillText(lab,inX-13,y0+k*gap));
      c.fillStyle=`rgb(${FG})`;c.font=font(10,true);c.textAlign='left';c.fillText(name.toUpperCase(),inX-13-c.measureText(name.toUpperCase()).width,y0-11);
    }else{
      const k=[t.skin,t.eye,t.shirt,t.pants,t.hair??0,t.hairC??0,t.cloth??0,t.shoe??3][g];
      c.fillStyle=`rgb(${MU})`;c.font=font(11);c.fillText(name,inX-13,(y0+y1)/2-5);
      c.fillStyle=`rgb(${FG})`;c.font=font(10);c.fillText(ACTIVE[g],inX-13,(y0+y1)/2+6);
      // a short tick from the label to the lit input
      c.strokeStyle=`rgba(${FG},.5)`;c.beginPath();c.moveTo(inX-11,(y0+y1)/2+6);c.lineTo(inX-6,y0+k*gap);c.stroke();
    }
    yy+=7;
  });
  const netB=yy-7,hidY=Array.from({length:HIDDEN},(_,j)=>top+6+j*((netB-top-12)/(HIDDEN-1)));
  const outY=Array.from({length:OUTS},(_,o)=>o===0?top+14:top+64+(o-1)*((netB-top-74)/(OUTS-2)));
  c.textAlign='center';c.fillStyle=`rgb(${MU})`;c.font=font(10);
  c.fillText('TRAITS',inX,top-14);c.fillText('HIDDEN NEURONS',hidX,top-14);c.fillText('VERDICT',outX+30,top-14);
  const act=m?m.act(x):null;
  if(m){
    let mx=1e-6;m.W1.forEach(r=>r.forEach(w=>mx=Math.max(mx,Math.abs(w))));
    for(let j=0;j<HIDDEN;j++)for(let k=0;k<x.length;k++){
      const w=m.W1[j][k],a=(.06+.94*Math.min(1,Math.abs(x[k])))*Math.abs(w)/mx;
      if(a<.05)continue;
      c.strokeStyle=`rgba(${w>0?ROT:BLD},${(a*.75).toFixed(3)})`;c.lineWidth=.5+a*1.6;
      c.beginPath();c.moveTo(inX,inY[k]);c.lineTo(hidX,hidY[j]);c.stroke();
    }
    let m2=1e-6;m.W2.forEach(r=>r.forEach(w=>m2=Math.max(m2,Math.abs(w))));
    for(let o=0;o<OUTS;o++)for(let j=0;j<HIDDEN;j++){
      const w=m.W2[o][j],a=Math.abs(w)/m2*(.15+.85*Math.abs(act.a[j]));
      if(a<.06)continue;
      // for Good, green pushes toward Good; for a part, red pushes toward "this part is wrong"
      const toward=w*act.a[j]>0,good=o===0?toward:!toward;
      c.strokeStyle=`rgba(${good?ROT:BLD},${(.08+a*.8).toFixed(3)})`;c.lineWidth=(o===0?1:.6)+a*(o===0?3.5:2);
      c.beginPath();c.moveTo(hidX,hidY[j]);c.lineTo(outX,outY[o]);c.stroke();
    }
  }
  inY.forEach((y,k)=>{const v=Math.min(1,Math.abs(x[k]));c.fillStyle=`rgba(${FG},${(.12+.88*v).toFixed(3)})`;c.beginPath();c.arc(inX,y,3,0,7);c.fill();});
  hidY.forEach((y,j)=>{
    const a=act?act.a[j]:0;
    c.fillStyle=`rgba(${a>=0?ROT:BLD},${(.12+.88*Math.abs(a)).toFixed(3)})`;c.strokeStyle=`rgba(${FG},.45)`;c.lineWidth=1;
    c.beginPath();c.arc(hidX,y,8,0,7);c.fill();c.stroke();
  });
  for(let o=0;o<OUTS;o++){
    const p=act?act.p[o]:(o?0:.5),y=outY[o],R=o===0?12:7;
    const bad=o===0?p<.5:p>=.5,strength=o===0?Math.abs(p-.5)*2:p;
    c.fillStyle=`rgba(${bad?BLD:ROT},${(.12+.88*strength).toFixed(3)})`;c.strokeStyle=`rgba(${FG},${o?.5:1})`;c.lineWidth=o?1:1.5;
    c.beginPath();c.arc(outX,y,R,0,7);c.fill();c.stroke();
    c.textAlign='left';
    if(o===0){c.fillStyle=`rgb(${FG})`;c.font='700 14px "Pixelify Sans",monospace';c.fillText(m?`GOOD ${Math.round(p*100)}%`:'GOOD ?',outX+18,y);}
    else{c.fillStyle=p>=.5?`rgb(${BLD})`:`rgb(${MU})`;c.font=font(11,p>=.5);c.fillText(`${TAGS[o-1]}${m?' '+Math.round(p*100)+'%':''}`,outX+13,y);}
  }
  c.textAlign='left';c.fillStyle=`rgb(${MU})`;c.font=font(10);c.fillText('WHAT IS WRONG',outX-6,top+42);

  // loss curve: how wrong the network still is, epoch by epoch
  const lx=40,lw=HW-56,ly=netB+34,lh=60;
  c.textAlign='left';c.fillStyle=`rgb(${MU})`;c.font=font(10);c.fillText('LEARNING · error per epoch',lx,ly-12);
  c.strokeStyle=`rgba(${MU},.35)`;c.lineWidth=1;c.beginPath();c.moveTo(lx,ly);c.lineTo(lx,ly+lh);c.lineTo(lx+lw,ly+lh);c.stroke();
  if(m&&m.loss.length){
    const L=m.loss,mxL=Math.max(...L.slice(0,3),.0001);
    c.strokeStyle=`rgb(${ROT})`;c.lineWidth=1.6;c.beginPath();
    L.forEach((v,i)=>{const px=lx+i/(EPOCHS-1)*lw,py=ly+lh-Math.min(1,v/mxL)*lh;i?c.lineTo(px,py):c.moveTo(px,py);});
    c.stroke();
    const lp=lx+(L.length-1)/(EPOCHS-1)*lw,lv=ly+lh-Math.min(1,L[L.length-1]/mxL)*lh;
    c.fillStyle=`rgb(${FG})`;c.beginPath();c.arc(lp,lv,3,0,7);c.fill();
    c.textAlign='right';c.fillStyle=`rgb(${FG})`;c.font=font(11);
    c.fillText(`epoch ${m.epoch}/${EPOCHS} · error ${L[L.length-1].toFixed(3)}`,lx+lw,ly-12);
  }else{c.textAlign='right';c.fillText('waiting for ratings',lx+lw,ly-12);}

  // smart roll: the 60 candidates it weighed
  const ty=ly+lh+36,th=36,bw=lw/60;
  c.textAlign='left';c.fillStyle=`rgb(${MU})`;c.font=font(10);c.fillText('THINKING · smart roll candidates',lx,ty-12);
  if(think){
    think.scores.forEach((v,i)=>{
      const isTop=think.top&&think.top.includes(i),isPick=think.pick===i;
      c.fillStyle=isTop?`rgb(${ROT})`:`rgba(${MU},.55)`;
      c.fillRect(lx+i*bw+1,ty+th-v*th,Math.max(1,bw-2),Math.max(1,v*th));
      if(isPick){c.strokeStyle=`rgb(${FG})`;c.lineWidth=1.5;c.strokeRect(lx+i*bw,ty-2,bw,th+4);}
    });
    if(think.pick!=null){c.textAlign='right';c.fillStyle=`rgb(${FG})`;c.font=font(11);c.fillText(`picked ${Math.round(think.scores[think.pick]*100)}% from the top 5`,lx+lw,ty-12);}
  }else{c.textAlign='right';c.fillText(net?'turn on Smart roll and roll':'available after training',lx+lw,ty-12);}

  document.getElementById('hudState').textContent=training?`learning · epoch ${training.epoch}`:net?(netStats.cv==null?'imported model':`trained · agrees ${pct(netStats.cv)}`):`needs ratings`;
  document.getElementById('hudFoot').textContent=(hudFocus?`Showing the hovered zombie #${t.seed}. `:`Showing the zombie in the editor #${t.seed}. `)+'Green lines push toward Good, red toward Bad or toward a part being wrong; brighter means a stronger, more active connection.';
}

