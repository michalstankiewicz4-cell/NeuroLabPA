// SDF Walkers v0.14.0 · Export and import of ratings and trained models.
// ---------- export / import ----------
// A model file carries its input list, so a file made for another version of the generator is refused, not misread.
function featureNames(){
  const one=[SKINS,EYES,CLOTH,CLOTH,HAIR_STYLES,HAIRS,GARMENTS,SHOES];
  const pre=['skin','eye glow','shirt','trousers','hair','hair colour','clothing','shoes'];
  const out=[];
  one.forEach((list,i)=>list.forEach(o=>out.push(pre[i]+': '+o.name)));
  ['height','build','shoulders','female','head size','jaw open','head tilt','hunch','limp','bare skull','arm gone','eye gone','shirt↔trousers hue gap','shirt↔skin hue gap'].forEach(n=>out.push(n));
  return out;
}
const ioNote=document.getElementById('ioNote');
const r5=a=>Array.isArray(a)?a.map(r5):Math.round(a*1e5)/1e5;
async function offerFile(filename,text){
  try{
    const dl=window.claude&&window.claude.use?await window.claude.use('downloads'):null;
    if(dl){await dl.save({filename,data:text});ioNote.textContent=`Saved ${filename}.`;return;}
  }catch(e){
    if(e&&e.code==='declined'){ioNote.textContent='Save cancelled.';return;}
  }
  if(!window.claude){   // standalone page (GitHub Pages, local server): a normal browser download works
    const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'application/json'}));a.download=filename;
    document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    ioNote.textContent=`Downloaded ${filename}.`;return;
  }
  try{await navigator.clipboard.writeText(text);ioNote.textContent=`This view cannot save files, so ${filename} was copied to the clipboard instead. Paste it into a new file.`;}
  catch(e){ioNote.textContent='This view can neither save files nor use the clipboard.';}
}
document.getElementById('expRatings').addEventListener('click',()=>{
  const rated=ratings.filter(r=>typeof r.good==='boolean');
  if(!rated.length){ioNote.textContent='There are no ratings to export yet.';return;}
  offerFile(`sdf-walkers-ratings-${rated.length}.json`,JSON.stringify({format:'sdf-walkers-ratings',version:2,exported:new Date().toISOString(),ratings:rated},null,1));
});
document.getElementById('expModel').addEventListener('click',()=>{
  if(!net){ioNote.textContent='The network has not been trained yet, so there is no model to export.';return;}
  const data={format:'sdf-walkers-model',version:1,created:new Date().toISOString(),
    trainedOn:netStats?netStats.n:0,inputs:featureNames(),outputs:['good',...TAGS],hidden:net.h,activation:'tanh',outputActivation:'sigmoid',
    W1:r5(net.W1),b1:r5(net.b1),W2:r5(net.W2),b2:r5(net.b2)};
  offerFile(`sdf-walkers-model-${data.trainedOn}.json`,JSON.stringify(data));
});
document.getElementById('impModel').addEventListener('change',e=>{
  const file=e.target.files&&e.target.files[0];e.target.value='';
  if(!file)return;
  const fr=new FileReader();
  fr.onload=()=>{
    let d;
    try{d=JSON.parse(fr.result);}catch(err){ioNote.textContent='That file is not valid JSON.';return;}
    if(d.format!=='sdf-walkers-model'){ioNote.textContent='That file is not an SDF Walkers model.';return;}
    const want=featureNames(),outs=['good',...TAGS];
    if(!Array.isArray(d.inputs)||d.inputs.length!==want.length||d.inputs.some((n,i)=>n!==want[i])||JSON.stringify(d.outputs)!==JSON.stringify(outs)){
      ioNote.textContent=`This model was made for a different version of the generator (${(d.inputs||[]).length} inputs, this page uses ${want.length}). Train it again from ratings instead.`;return;
    }
    const m=createNet(want.length,d.hidden,1);
    Object.assign(m,{W1:d.W1,b1:d.b1,W2:d.W2,b2:d.b2,epoch:EPOCHS,loss:[]});
    trainToken++;training=null;net=m;
    netStats={...(netStats||{}),imported:true,importedFrom:d.trainedOn,cv:null};
    netStats.insights=insights(ratings.filter(r=>typeof r.good==='boolean'));
    renderModel();refreshScores();drawHud();
    ioNote.textContent=`Loaded a model trained on ${d.trainedOn} ratings. Your next rating retrains the network from this page's ratings and replaces it.`;
  };
  fr.readAsText(file);
});

