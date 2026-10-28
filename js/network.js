// SDF Walkers v0.15.0 · The rating network: features, multi-output MLP, training, cross-validation, insights, model panel.
// ---------- ratings: the network ----------
// Input: the specimen's own traits plus two colour relations. Output: chance you rate it Good.
const FEAT_N=55;
function feats(t){
  const v=[];const oh=(i,n)=>{for(let k=0;k<n;k++)v.push(k===i?1:0);};
  oh(t.skin,4);oh(t.eye,3);oh(t.shirt,6);oh(t.pants,6);oh(t.hair??0,4);oh(t.hairC??0,4);oh(t.cloth??0,9);oh(t.shoe??3,5);
  v.push(((t.height??1)-.88)/.24,t.build??.5,t.shoulder??.5,t.sex??0,(t.head-.16)/.14,(t.jaw-.06)/.14,Math.abs(t.tilt)/.3,(t.hunch-.08)/.42,t.limp/.85,
    t.boneR?1:0,t.armGone?1:0,(t.eyeL&&t.eyeR)?0:1);
  v.push(hueDist(CLOTH[t.shirt].h,CLOTH[t.pants].h)/180,hueDist(CLOTH[t.shirt].h,SKINS[t.skin].h)/180);
  return v;
}
const EPOCHS=400,HIDDEN=12,OUTS=1+TAGS.length;   // output 0: Good; outputs 1..6: "this part is wrong"
function createNet(n,h,seed){
  const K=OUTS,r=mulberry(seed),rn=()=>(r()-.5)*.6;
  const m={n,h,K,W1:Array.from({length:h},()=>Array.from({length:n},rn)),b1:new Array(h).fill(0),
    W2:Array.from({length:K},()=>Array.from({length:h},rn)),b2:new Array(K).fill(0),epoch:0,loss:[]};
  m.act=x=>{
    const a=new Array(h);
    for(let j=0;j<h;j++){let z=m.b1[j];for(let k=0;k<n;k++)z+=m.W1[j][k]*x[k];a[j]=Math.tanh(z);}
    const p=new Array(K);
    for(let o=0;o<K;o++){let z=m.b2[o];for(let j=0;j<h;j++)z+=m.W2[o][j]*a[j];p[o]=1/(1+Math.exp(-z));}
    return{a,p};
  };
  m.predict=x=>m.act(x).p[0];
  // Y: targets per output, M: 1 where that target is known (a Bad with no tags says nothing about parts)
  m.step=(X,Y,M,E,Wt)=>{
    const lr=.15,l2=.002,N=X.length;
    for(let e=0;e<E;e++){
      const gW1=m.W1.map(row=>row.map(()=>0)),gb1=new Array(h).fill(0),gW2=m.W2.map(row=>row.map(()=>0)),gb2=new Array(K).fill(0);let L=0;
      for(let s=0;s<N;s++){
        const {a,p}=m.act(X[s]),x=X[s],da=new Array(h).fill(0),w=Wt?Wt[s]:1;
        for(let o=0;o<K;o++){
          if(!M[s][o])continue;
          const d=(p[o]-Y[s][o])*w;
          L-=w*(Y[s][o]?Math.log(p[o]+1e-9):Math.log(1-p[o]+1e-9));
          gb2[o]+=d;
          for(let j=0;j<h;j++){gW2[o][j]+=d*a[j];da[j]+=d*m.W2[o][j];}
        }
        for(let j=0;j<h;j++){const dz=da[j]*(1-a[j]*a[j]);gb1[j]+=dz;for(let k=0;k<n;k++)gW1[j][k]+=dz*x[k];}
      }
      for(let j=0;j<h;j++){m.b1[j]-=lr*gb1[j]/N;for(let k=0;k<n;k++)m.W1[j][k]-=lr*(gW1[j][k]/N+l2*m.W1[j][k]);}
      for(let o=0;o<K;o++){m.b2[o]-=lr*gb2[o]/N;for(let j=0;j<h;j++)m.W2[o][j]-=lr*(gW2[o][j]/N+l2*m.W2[o][j]);}
      m.epoch++;m.loss.push(L/N);
    }
  };
  return m;
}
function trainNet(X,Y,M,seed=1,Wt){const m=createNet(X[0].length,HIDDEN,seed);m.step(X,Y,M,EPOCHS,Wt);return m;}
// Ratings carry who made them. Claude's count less than yours when both are used.
const CLAUDE_WEIGHT=.6;
const isClaude=r=>r.rater==='claude';
function sourceFilter(list){const src=document.getElementById('trainOn').value;return src==='both'?list:list.filter(r=>src==='claude'?isClaude(r):!isClaude(r));}
let retrainTimer=0;
function scheduleRetrain(){clearTimeout(retrainTimer);retrainTimer=setTimeout(()=>{cvCache=null;retrain();},400);}
let net=null,netStats=null;
const MIN_TOTAL=16,MIN_EACH=4;
let trainToken=0,training=null,cvCache=null;
function retrain(){
  const rated=sourceFilter(ratings.filter(r=>typeof r.good==='boolean'));
  const src=document.getElementById('trainOn').value;
  const Wt=rated.map(r=>src==='both'&&isClaude(r)?CLAUDE_WEIGHT:1);
  const pos=rated.filter(r=>r.good).length,neg=rated.length-pos;
  if(!(rated.length>=MIN_TOTAL&&pos>=MIN_EACH&&neg>=MIN_EACH)){
    net=null;netStats={n:rated.length,pos,neg};renderModel();refreshScores();drawHud();return;
  }
  const X=rated.map(r=>feats(r.traits)),y=rated.map(r=>r.good?1:0);
  const Y=rated.map(r=>[r.good?1:0,...TAGS.map(t=>(r.tags||[]).includes(t)?1:0)]);
  const M=rated.map(r=>[1,...TAGS.map(()=>r.good||(r.tags&&r.tags.length)?1:0)]);
  // 5-fold cross-validation: how often the network agrees with a rating it has not seen
  const idx=X.map((_,i)=>i),r=mulberry(99);
  for(let i=idx.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
  // re-measured every 5 new ratings so a click never waits on 5 extra trainings
  if(!cvCache||rated.length-cvCache.n>=5){
    let hit=0;
    for(let f=0;f<5;f++){
      const test=idx.filter((_,i)=>i%5===f),train=idx.filter((_,i)=>i%5!==f);
      const m=trainNet(train.map(i=>X[i]),train.map(i=>Y[i]),train.map(i=>M[i]),f+2,train.map(i=>Wt[i]));
      test.forEach(i=>{if((m.predict(X[i])>.5)===(y[i]===1))hit++;});
    }
    cvCache={n:rated.length,cv:hit/X.length};
  }
  const stats={n:rated.length,pos,neg,cv:cvCache.cv,base:Math.max(pos,neg)/X.length};
  const tok=++trainToken,m=createNet(FEAT_N,HIDDEN,1);
  const finish=()=>{
    if(tok!==trainToken)return;
    training=null;net=m;netStats=stats;netStats.insights=insights(rated);
    renderModel();refreshScores();drawHud();
  };
  if(hudOpen()&&!reduced){
    training=m;
    const stepF=()=>{if(tok!==trainToken)return;m.step(X,Y,M,Math.min(6,EPOCHS-m.epoch),Wt);drawHud();if(m.epoch<EPOCHS)requestAnimationFrame(stepF);else finish();};
    requestAnimationFrame(stepF);
  }else{m.step(X,Y,M,EPOCHS,Wt);finish();}
}
function effects(rated,o){
  const T=rated.length>=10?rated.map(r=>r.traits):Array.from({length:150},(_,i)=>ser(makeZombie(900001+i)));
  const avg=mod=>T.reduce((s,t)=>s+net.act(feats(mod({...t}))).p[o],0)/T.length;
  const base=avg(t=>t),out=[];
  const cat=(label,key,list)=>list.forEach((x,i)=>out.push({label:`${label}: ${x.name}`,d:avg(t=>(t[key]=i,t))-base}));
  cat('Skin','skin',SKINS);cat('Eyes','eye',EYES);cat('Shirt','shirt',CLOTH);cat('Trousers','pants',CLOTH);
  cat('Hair','hair',HAIR_STYLES);cat('Hair colour','hairC',HAIRS);cat('Clothing','cloth',GARMENTS);cat('Shoes','shoe',SHOES);
  out.push({label:'Female',d:avg(t=>(t.sex=1,t))-avg(t=>(t.sex=0,t))});
  out.push({label:'Tall',d:avg(t=>(t.height=1.1,t))-avg(t=>(t.height=.9,t))});
  out.push({label:'Heavy build',d:avg(t=>(t.build=.9,t))-avg(t=>(t.build=.1,t))});
  out.push({label:'Broad shoulders',d:avg(t=>(t.shoulder=.9,t))-avg(t=>(t.shoulder=.1,t))});
  out.push({label:'Exposed skull',d:avg(t=>(t.boneR=.3,t))-avg(t=>(t.boneR=0,t))});
  out.push({label:'Missing arm',d:avg(t=>(t.armGone=1,t))-avg(t=>(t.armGone=0,t))});
  out.push({label:'Limp',d:avg(t=>(t.limp=.7,t))-avg(t=>(t.limp=0,t))});
  out.push({label:'Bigger head (+30%)',d:avg(t=>(t.head=.26,t))-avg(t=>(t.head=.2,t))});
  out.push({label:'Heavy hunch',d:avg(t=>(t.hunch=.45,t))-avg(t=>(t.hunch=.15,t))});
  out.push({label:'Wide-open jaw',d:avg(t=>(t.jaw=.18,t))-avg(t=>(t.jaw=.08,t))});
  return out.sort((a,b)=>b.d-a.d);
}
function insights(rated){
  const g=effects(rated,0);
  const parts=TAGS.map((t,i)=>{
    const marks=rated.filter(r=>(r.tags||[]).includes(t)).length||(rated.length?0:3);
    return{tag:t,marks,causes:marks>=3?effects(rated,i+1).slice(0,2).filter(o=>o.d>.04):[]};
  });
  return{likes:g.slice(0,4).filter(o=>o.d>.02),dislikes:g.slice(-4).reverse().filter(o=>o.d<-.02),parts};
}
const pct=v=>Math.round(v*100)+'%';
function partsHtml(parts){
  const rows=parts.filter(p=>p.marks).map(p=>`<li><span>${p.tag} <span class="note">×${p.marks}</span></span><span class="down">${
    p.marks<3?'needs 3 marks':(p.causes.length?p.causes.map(c=>c.label).join(', '):'no clear trait')}</span></li>`);
  return rows.length?`<ul>${rows.join('')}</ul>`:'<p class="note">No parts marked yet.</p>';
}
function renderModel(){
  const s=netStats||{n:0,pos:0,neg:0},box=document.getElementById('modelbox'),smart=document.getElementById('smart');
  const where=store?(store.kind==='db'?'Ratings are saved with this page.':'Ratings are saved in this browser only.'):'';
  const tagCounts=TAGS.map(t=>[t,ratings.filter(r=>r.tags&&r.tags.includes(t)).length]).filter(x=>x[1]).sort((a,b)=>b[1]-a[1]);
  const tagHtml=tagCounts.length?`<ul>${tagCounts.map(([t,c])=>`<li><span>${t}</span><span>${c}</span></li>`).join('')}</ul>`:'<p class="note">No parts marked yet.</p>';
  if(!net){
    const need=Math.max(MIN_TOTAL-s.n,0),needE=Math.max(MIN_EACH-s.pos,0)+Math.max(MIN_EACH-s.neg,0);
    const msg=need||needE?`Rate ${Math.max(need,needE)} more to train the network${needE?` (it needs at least ${MIN_EACH} Good and ${MIN_EACH} Bad)`:''}.`:'Training…';
    box.innerHTML=`<div><h3>Network</h3><p class="note">${msg} ${where}</p><div class="meter"><i style="width:${Math.min(100,s.n/MIN_TOTAL*100)}%"></i></div></div>
      <div><h3>Rated so far</h3><ul><li><span>Good</span><span>${s.pos}</span></li><li><span>Bad</span><span>${s.neg}</span></li></ul></div>
      <div><h3>Parts marked wrong</h3>${tagHtml}</div>`;
    smart.disabled=true;smart.checked=false;
  }else{
    const ins=s.insights;
    const li=(arr,cls)=>arr.length?`<ul>${arr.map(o=>`<li><span>${o.label}</span><span class="${cls}">${o.d>0?'+':''}${Math.round(o.d*100)}</span></li>`).join('')}</ul>`:'<p class="note">Nothing stands out yet.</p>';
    const head=s.imported?`Imported model, trained on ${s.importedFrom} ratings elsewhere. It has not been tested against this page's ratings.`:`Trained on ${s.n} ratings (${s.pos} Good, ${s.neg} Bad). On ratings it has not seen it agrees with you ${pct(s.cv)} of the time; always guessing the more common answer would score ${pct(s.base)}.`;
    box.innerHTML=`<div><h3>Network</h3><p class="note">${head} ${where}</p></div>
      <div><h3>Raises your rating</h3>${li(ins.likes,'up')}</div>
      <div><h3>Lowers your rating</h3>${li(ins.dislikes,'down')}</div>
      <div><h3>Why parts get marked</h3>${partsHtml(ins.parts)}</div>`;
    smart.disabled=false;
  }
  const all=ratings.filter(r=>typeof r.good==='boolean'),cl=all.filter(isClaude).length;
  document.getElementById('rateStatus').textContent=`Good means every part is fine. If something is wrong, mark Bad and the parts that are wrong. Rated so far: ${all.length-cl} by you, ${cl} by Claude.`;
}

