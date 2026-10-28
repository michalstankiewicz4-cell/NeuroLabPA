// SDF Walkers v0.16.1 · Animated four-direction preview.
// ---------- preview ----------
const pv=document.getElementById('preview'),ctx=pv.getContext('2d');
let sheetBitmap=null,tick=0,last=0,cur=0;
const animEl=document.getElementById('anim');
const tabs=document.getElementById('tabs');
ANIMS.forEach((an,i)=>{
  const b=document.createElement('button');b.type='button';b.textContent=an.label;b.id='tab-'+an.key;
  b.setAttribute('aria-pressed',i===0?'true':'false');
  b.addEventListener('click',()=>{cur=i;tick=0;[...tabs.children].forEach((c,j)=>c.setAttribute('aria-pressed',j===i?'true':'false'));});
  tabs.appendChild(b);
});
function frameIndex(){
  const an=ANIMS[cur];
  if(an.loop)return tick%FRAMES;
  const hold=Math.round(an.fps*.8),n=tick%(FRAMES+hold);   // one-shot clips hold their last pose, then replay
  return Math.min(n,FRAMES-1);
}
function loop(t){
  const an=ANIMS[cur];
  if(animEl.checked&&t-last>1000/an.fps){tick++;last=t;}
  if(sheetBitmap){
    if(pv.width!==FW*4||pv.height!==FH){pv.width=FW*4;pv.height=FH;}
    ctx.imageSmoothingEnabled=false;
    ctx.clearRect(0,0,pv.width,pv.height);
    const k=frameIndex();
    for(let d=0;d<4;d++)ctx.drawImage(sheetBitmap,k*FW,(cur*DIRS.length+d)*FH,FW,FH,d*FW,0,FW,FH);
  }
  requestAnimationFrame(loop);
}

