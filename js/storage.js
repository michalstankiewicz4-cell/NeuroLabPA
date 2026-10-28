// SDF Walkers v0.16.0 · Where ratings live: the page database for the owner on claude.ai, browser storage elsewhere.
// ---------- ratings: storage ----------
// The owner's ratings live with the page (db); anyone else keeps them in this browser only.
// Tags only go with Bad: they say which part is wrong. Good means every part is fine.
const TAGS=['Head','Torso','Arms','Legs','Skin colour','Clothes colour'];
let ratings=[],store=null;
const LS='sdf-walkers-ratings-v2';
let storeReady=Promise.resolve();
const initStore=async()=>{
  try{
    const c=window.claude;
    if(c&&c.use){
      const [db,user]=await Promise.all([c.use('db'),c.use('user')]);
      if(db&&user&&await user.isOwner()){
        store={kind:'db',col:db.collection('ratings')};
        // subscribe once: ratings written elsewhere (another tab, or Claude rating a batch) arrive live and retrain the network
        await new Promise(res=>{
          let first=true;
          store.col.limit(1000).onSnapshot(snap=>{
            ratings=snap.docs.map(d=>d.data()).filter(r=>r.v===2);
            if(first){first=false;res();}else scheduleRetrain();
          },()=>{if(first){first=false;res();}});
        });
      }
    }
  }catch(e){store=null;}
  if(!store){
    store={kind:'local'};
    try{localStorage.removeItem('sdf-walkers-ratings');ratings=JSON.parse(localStorage.getItem(LS)||'[]');}catch(e){ratings=[];}
  }
  retrain();
};
function saveLocal(){try{localStorage.setItem(LS,JSON.stringify(ratings));}catch(e){}}
async function persist(rec){
  await storeReady;
  const i=ratings.findIndex(r=>r.id===rec.id);
  if(i<0)ratings.push(rec);else ratings[i]=rec;
  if(store.kind==='db'){
    try{await store.col.doc(rec.id).set(rec);}
    catch(e){store={kind:'local'};saveLocal();}
  }else saveLocal();
  retrain();
}

