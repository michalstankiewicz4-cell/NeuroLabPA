// SDF Walkers v0.17.0 · Wiring and start-up.
document.getElementById('copy').addEventListener('click',e=>{
  const btn=e.currentTarget,txt=document.getElementById('json').textContent;
  navigator.clipboard.writeText(txt).then(()=>{btn.textContent='Copied';setTimeout(()=>btn.textContent='Copy JSON',1500);})
  .catch(()=>{const r=document.createRange();r.selectNodeContents(document.getElementById('json'));const s=getSelection();s.removeAllRanges();s.addRange(r);btn.textContent='Selected, press Ctrl+C';});
});

document.getElementById('appVer').textContent=APP_VERSION;

if(!gl)fail('This browser does not support WebGL2, so the shader cannot start.');
else (async()=>{
  try{
    showLoader('Compiling shader',null,'');
    const ms=await setup((t,f,n)=>showLoader(t,f,n));
    console.info(`SDF Walkers ${APP_VERSION}: shaders ready in ${ms} ms`);
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
  }catch(e){hideLoader();fail('Shader error: '+e.message);}
})();
