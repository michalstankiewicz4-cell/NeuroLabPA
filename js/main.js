// SDF Walkers v0.16.0 · Wiring and start-up.
document.getElementById('copy').addEventListener('click',e=>{
  const btn=e.currentTarget,txt=document.getElementById('json').textContent;
  navigator.clipboard.writeText(txt).then(()=>{btn.textContent='Copied';setTimeout(()=>btn.textContent='Copy JSON',1500);})
  .catch(()=>{const r=document.createRange();r.selectNodeContents(document.getElementById('json'));const s=getSelection();s.removeAllRanges();s.addRange(r);btn.textContent='Selected, press Ctrl+C';});
});

function rebuild(){try{renderSheet();}catch(e){fail('Render error: '+e.message);}}

document.getElementById('appVer').textContent=APP_VERSION;

if(!gl)fail('This browser does not support WebGL2, so the shader cannot start.');
else{
  try{
    setup();
    Z=makeZombie(4417);syncEditor();rebuild();
    window.hudReady=true;
    newBatch();renderModel();
    storeReady=initStore();
    let hudSaved=false;try{hudSaved=localStorage.getItem('sdf-walkers-hud')==='1';}catch(e){}
    initHudDrag();
    setHud(hudSaved);
    hudOnEl.addEventListener('change',()=>setHud(hudOnEl.checked));
    document.getElementById('hudBtn').addEventListener('click',()=>setHud(hud.hidden));
    document.getElementById('hudClose').addEventListener('click',()=>setHud(false));
    if(reduced)animEl.checked=false;
    document.getElementById('roll').addEventListener('click',roll);
    document.getElementById('batch').addEventListener('click',newBatch);
    document.getElementById('trainOn').addEventListener('change',()=>{cvCache=null;retrain();});
    ['elev','outline','polish'].forEach(id=>document.getElementById(id).addEventListener('change',()=>{rebuild();newBatch();}));
    document.getElementById('size').addEventListener('change',e=>{[FW,FH]=e.target.value.split('x').map(Number);rebuild();newBatch();});
    requestAnimationFrame(loop);
  }catch(e){fail('Shader error: '+e.message);}
}
