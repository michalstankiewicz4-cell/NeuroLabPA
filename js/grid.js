// SDF Walkers v0.17.0 · The rating grid.
// ---------- ratings: the grid ----------
const grid=document.getElementById('grid');
let cards=[];
function newBatch(){
  grid.textContent='';cards=[];
  for(let i=0;i<12;i++){
    const sd=1+Math.floor(Math.random()*99999),z=makeZombie(sd);
    const card={z,rec:null,tags:new Set(),id:'r'+Date.now().toString(36)+'-'+sd,pending:Promise.resolve()};
    const el=document.createElement('div');el.className='zc';
    const cv=document.createElement('canvas');cv.className='pix';cv.width=FW*2;cv.height=FH;
    cv.setAttribute('aria-label','Zombie '+sd+', front and side view');
    cv.classList.add('pending');
    glJob(()=>{drawFrames(z,[[0,0,0],[0,2,.25]],2);cv.getContext('2d').drawImage(glc,0,0);cv.classList.remove('pending');});
    const cap=document.createElement('div');cap.className='cap';
    cap.innerHTML=`<span>#${sd}</span><span class="score"></span>`;
    const vote=document.createElement('div');vote.className='vote';
    const g=document.createElement('button');g.type='button';g.className='g';g.textContent='Good';g.setAttribute('aria-pressed','false');
    const b=document.createElement('button');b.type='button';b.className='b';b.textContent='Bad';b.setAttribute('aria-pressed','false');
    const op=document.createElement('button');op.type='button';op.textContent='Open';op.title='Load into the editor';
    vote.append(g,b,op);
    const tr=document.createElement('div');tr.className='tagrow';
    TAGS.forEach(t=>{
      const tb=document.createElement('button');tb.type='button';tb.textContent=t;tb.setAttribute('aria-pressed','false');
      tb.addEventListener('click',()=>{
        card.tags.has(t)?card.tags.delete(t):card.tags.add(t);
        tb.setAttribute('aria-pressed',card.tags.has(t)?'true':'false');
        if(!card.rec||card.rec.good)setVote(false);else save(card);   // marking a part means Bad
      });
      tr.appendChild(tb);
    });
    const setVote=v=>{
      if(v){card.tags.clear();tr.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed','false'));}   // Good means every part is fine
      card.rec={v:2,rater:'you',id:card.id,seed:sd,traits:ser(z),good:v,t:Date.now()};
      g.setAttribute('aria-pressed',v?'true':'false');b.setAttribute('aria-pressed',v?'false':'true');
      el.classList.toggle('good',v);el.classList.toggle('bad',!v);
      save(card);
    };
    g.addEventListener('click',()=>setVote(true));
    b.addEventListener("click",()=>setVote(false));
    op.addEventListener('click',()=>{Z=des(ser(z));syncEditor();rebuild();document.querySelector('.view').scrollIntoView({behavior:'smooth',block:'start'});});
    el.append(cv,cap,vote,tr);grid.appendChild(el);
    card.el=el;card.scoreEl=cap.querySelector('.score');cards.push(card);
    const focusOn=()=>{hudFocus=ser(z);drawHud();},focusOff=()=>{hudFocus=null;drawHud();};
    el.addEventListener('mouseenter',focusOn);el.addEventListener('mouseleave',focusOff);
    el.addEventListener('focusin',focusOn);el.addEventListener('focusout',focusOff);
  }
  refreshScores();
}
function save(card){
  const rec={...card.rec,tags:[...card.tags]};
  card.pending=card.pending.then(()=>persist(rec));   // one write at a time per rating
}
function refreshScores(){
  cards.forEach(c=>{
    if(!net){c.scoreEl.textContent='';return;}
    const p=net.act(feats(ser(c.z))).p;let best=-1;
    for(let i=1;i<p.length;i++)if(p[i]>.5&&(best<0||p[i]>p[best]))best=i;
    c.scoreEl.textContent='NET '+pct(p[0])+(best>0?' · '+TAGS[best-1].toUpperCase()+'?':'');
  });
}

