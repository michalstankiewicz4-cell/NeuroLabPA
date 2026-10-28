// SDF Walkers v0.16.2 · WebGL setup, frame rendering, sprite sheet and its JSON.
// ---------- WebGL ----------
const glc=document.createElement('canvas');
const gl=glc.getContext('webgl2',{antialias:false,premultipliedAlpha:false,preserveDrawingBuffer:true});
const errEl=document.getElementById('err');
function fail(m){errEl.textContent=m;errEl.hidden=false;}
// Shaders are compiled without blocking the page where the browser allows it (KHR_parallel_shader_compile):
// on Windows the translation to DirectX can take seconds, and a blocking compile freezes the tab with nothing in the console.
function startProgram(fs){
  const p=gl.createProgram(),mk=(type,src)=>{const sh=gl.createShader(type);gl.shaderSource(sh,src);gl.compileShader(sh);gl.attachShader(p,sh);return sh;};
  p._shaders=[mk(gl.VERTEX_SHADER,VS),mk(gl.FRAGMENT_SHADER,fs)];
  gl.bindAttribLocation(p,0,'a');gl.linkProgram(p);return p;
}
function checkProgram(p){
  if(gl.getProgramParameter(p,gl.LINK_STATUS))return p;
  const logs=p._shaders.map(sh=>gl.getShaderInfoLog(sh)).filter(Boolean).join('\n');
  throw new Error((logs||gl.getProgramInfoLog(p)||'link failed').slice(0,400));
}
function prog(fs){return checkProgram(startProgram(fs));}
const N1=['uOff','uFrame','uPhase','uYaw','uElev','uScale','uAnim','uPolish','uHead','uTilt','uJaw','uEyeL','uEyeR','uBoneR','uBoneX','uMissing','uWound','uHunch','uArmGone','uLimp','uHS','uBuild','uShoulder','uHair','uCloth','uSex','uShoe','uArmShadow','uZero'];
const N2=['uTex','uCell','uPal','uOut','uShadow','uOutline','uPolish'];
let P1,P2,U1={},U2={},fb,tex;
let FW=64,FH=80,Z;
const nextFrame=()=>new Promise(r=>requestAnimationFrame(()=>r()));
// Draw calls only queue work; a fence tells us when the GPU has actually finished it.
// Polling it once per frame keeps the page responsive and makes the progress bar honest.
async function gpuDone(){
  const sync=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);
  gl.flush();
  if(!sync)return nextFrame();
  for(;;){
    const r=gl.clientWaitSync(sync,0,0);
    if(r===gl.ALREADY_SIGNALED||r===gl.CONDITION_SATISFIED||r===gl.WAIT_FAILED)break;
    await nextFrame();
  }
  gl.deleteSync(sync);
}
// Loader over the preview. Short jobs (an edit re-rendering the sheet) finish before the delay and never show it.
const loaderEl=document.getElementById('loader'),loaderBar=document.getElementById('loaderBar');
let loaderTimer=0;
function showLoader(title,frac,note,delay=0){
  document.getElementById('loaderTitle').textContent=title;
  document.getElementById('loaderNote').textContent=note||'';
  loaderBar.classList.toggle('busy',frac==null);
  loaderBar.firstElementChild.style.width=frac==null?'':Math.round(frac*100)+'%';
  if(loaderEl.hidden&&!loaderTimer)loaderTimer=setTimeout(()=>{loaderTimer=0;loaderEl.hidden=false;},delay);
}
function hideLoader(){clearTimeout(loaderTimer);loaderTimer=0;loaderEl.hidden=true;}

async function setup(onStatus){
  const ext=gl.getExtension('KHR_parallel_shader_compile');
  const t0=performance.now();
  const p1=startProgram(FS1),p2=startProgram(FS2);
  if(ext){
    while(!gl.getProgramParameter(p1,ext.COMPLETION_STATUS_KHR)||!gl.getProgramParameter(p2,ext.COMPLETION_STATUS_KHR)){
      const s=Math.round((performance.now()-t0)/1000);
      onStatus('Compiling shader',null,s>=3?`${s} s. The first start on Windows can take a while; later starts reuse the compiled shader.`:'');
      await new Promise(r=>setTimeout(r,100));
    }
  }else{onStatus('Compiling shader',null,'The page may pause for a moment.');await nextFrame();await nextFrame();}
  P1=checkProgram(p1);P2=checkProgram(p2);
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  N1.forEach(n=>U1[n]=gl.getUniformLocation(P1,n));N2.forEach(n=>U2[n]=gl.getUniformLocation(P2,n));
  fb=gl.createFramebuffer();tex=gl.createTexture();
  return Math.round(performance.now()-t0);
}
const ROWS=ANIMS.length*DIRS.length;
const polishEl=document.getElementById('polish');

// Every GPU job (sheet, thumbnails) runs through one queue, so a job that pauses between
// frames never finds the canvas resized or the uniforms changed by another job.
let glQueue=Promise.resolve();
function glJob(fn){glQueue=glQueue.then(fn).catch(e=>fail('Render error: '+e.message));return glQueue;}

function prepFrames(z,cols,rows){
  const W=FW*cols,H=FH*rows;
  glc.width=W;glc.height=H;
  gl.bindTexture(gl.TEXTURE_2D,tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,W,H,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.bindFramebuffer(gl.FRAMEBUFFER,fb);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
  gl.viewport(0,0,W,H);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(P1);
  const f=(n,v)=>gl.uniform1f(U1[n],v);
  gl.uniform1i(U1.uZero,0);
  gl.uniform2f(U1.uFrame,FW,FH);
  f('uElev',(+document.getElementById('elev').value)*Math.PI/180);f('uScale',1.38);
  f('uPolish',polishEl.checked?1:0);f('uHead',z.head);
  f('uTilt',z.tilt);f('uJaw',z.jaw);f('uEyeL',z.eyeL);f('uEyeR',z.eyeR);f('uBoneR',z.boneR);f('uBoneX',z.boneX);
  f('uMissing',z.missing);f('uWound',z.wound);f('uHunch',z.hunch);f('uArmGone',z.armGone);f('uLimp',z.limp);
  f('uHS',z.height);f('uBuild',z.build);f('uShoulder',z.shoulder);f('uHair',z.hair);f('uCloth',z.cloth);f('uSex',z.sex);f('uShoe',z.shoe);f('uArmShadow',window.ARM_SHADOW?1:0);
  return {W,H};
}
function pass1(list,cols,rows,from,to){
  gl.bindFramebuffer(gl.FRAMEBUFFER,fb);gl.useProgram(P1);
  for(let i=from;i<to;i++){
    const [ai,di,ph]=list[i],x=(i%cols)*FW,y=(rows-1-Math.floor(i/cols))*FH;
    gl.viewport(x,y,FW,FH);gl.uniform2f(U1.uOff,x,y);
    gl.uniform1f(U1.uAnim,ai);gl.uniform1f(U1.uYaw,DIRS[di].yaw);gl.uniform1f(U1.uPhase,ph);
    gl.drawArrays(gl.TRIANGLES,0,3);
  }
}
function pass2(z,W,H){
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,W,H);
  gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(P2);
  gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tex);gl.uniform1i(U2.uTex,0);
  gl.uniform2f(U2.uCell,FW,FH);gl.uniform3fv(U2.uPal,paletteArray(z));
  gl.uniform3f(U2.uOut,14/255,10/255,20/255);gl.uniform3f(U2.uShadow,10/255,8/255,16/255);
  gl.uniform1i(U2.uOutline,document.getElementById('outline').checked?1:0);
  gl.uniform1i(U2.uPolish,polishEl.checked?1:0);
  gl.drawArrays(gl.TRIANGLES,0,3);
}
// Render a list of frames [animIndex, dirIndex, phase] for specimen z, `cols` per row, in one go (small jobs).
function drawFrames(z,list,cols){
  const rows=Math.ceil(list.length/cols),{W,H}=prepFrames(z,cols,rows);
  pass1(list,cols,rows,0,list.length);pass2(z,W,H);
  return {W,H};
}
let sheetToken=0;
function rebuild(){const tok=++sheetToken;return glJob(()=>renderSheet(tok));}
async function renderSheet(tok){
  if(tok!==sheetToken)return;              // a newer request is already queued
  const list=[];
  ANIMS.forEach((an,ai)=>DIRS.forEach((d,di)=>{
    // looping clips wrap around; one-shot clips run from first pose to last
    for(let k=0;k<FRAMES;k++)list.push([ai,di,an.loop?k/FRAMES:k/(FRAMES-1)]);
  }));
  const z=Z,rows=Math.ceil(list.length/FRAMES),{W,H}=prepFrames(z,FRAMES,rows),first=!sheetBitmap;
  // one animation (4 rows) per browser frame, so the page stays responsive while the sheet renders
  const step=FRAMES*DIRS.length;
  for(let i=0;i<list.length;i+=step){
    pass1(list,FRAMES,rows,i,Math.min(list.length,i+step));
    await gpuDone();                        // wait for the GPU to really finish this chunk, without blocking the page
    const done=Math.min(list.length,i+step);
    showLoader('Rendering sprite sheet',done/list.length,`${done} of ${list.length} frames`,first?0:400);
    await nextFrame();
    if(tok!==sheetToken)return;
  }
  pass2(z,W,H);
  const url=glc.toDataURL('image/png');
  const img=document.getElementById('sheetImg');
  img.src=url;img.style.width=(W*2)+'px';
  const bm=new Image();bm.onload=()=>{sheetBitmap=bm;};bm.src=url;
  hideLoader();
  document.getElementById('sizeLabel').textContent=FW+'×'+FH+' PX / FRAME';
  document.getElementById('sheetNote').textContent=`Sheet ${W}×${H} px with a transparent background: ${ANIMS.length} animations × ${DIRS.length} directions = ${ROWS} rows of ${FRAMES} frames. Right-click or long-press the sheet, then choose “Save image”.`;
  writeJson(W,H);
}
function writeJson(W,H){
  const animations={};
  ANIMS.forEach((an,ai)=>{
    const rows={};DIRS.forEach((d,di)=>rows[d.key]=ai*DIRS.length+di);
    animations[an.key]={frames:FRAMES,fps:an.fps,loop:an.loop,rows};
  });
  const data={image:`zombie_${Z.seed}.png`,sheetWidth:W,sheetHeight:H,frameWidth:FW,frameHeight:FH,
    columns:FRAMES,origin:{x:FW/2,y:Math.round(FH*.93)},seed:Z.seed,animations};
  document.getElementById('json').textContent=JSON.stringify(data,null,2);
}

