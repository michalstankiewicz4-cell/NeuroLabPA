// SDF Walkers v0.17.0 · Rolling a new zombie, optionally steered by the network.
// ---------- rolling ----------
function roll(){
  const smart=document.getElementById('smart');
  if(net&&smart.checked){
    // draw 60 candidates, score each, pick one of the five the network likes best
    const pool=[];
    for(let i=0;i<60;i++){const z=makeZombie(1+Math.floor(Math.random()*99999));pool.push([net.predict(feats(ser(z))),z]);}
    const pickNow=()=>{
      const top=[...pool].sort((a,b)=>b[0]-a[0]).slice(0,5),pick=top[Math.floor(Math.random()*top.length)];
      think={scores:pool.map(p=>p[0]),top:top.map(t=>pool.indexOf(t)),pick:pool.indexOf(pick)};
      hudFocus=null;Z=pick[1];syncEditor();rebuild();drawHud();
    };
    if(hudOpen()&&!reduced){
      let i=0;
      const stepT=()=>{i=Math.min(pool.length,i+2);think={scores:pool.slice(0,i).map(p=>p[0])};hudFocus=ser(pool[i-1][1]);drawHud();
        if(i<pool.length)requestAnimationFrame(stepT);else pickNow();};
      requestAnimationFrame(stepT);
    }else pickNow();
  }else{Z=makeZombie(1+Math.floor(Math.random()*99999));syncEditor();rebuild();}
}

