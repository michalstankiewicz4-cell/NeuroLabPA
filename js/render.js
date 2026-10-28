// SDF Walkers v0.16.0 · WebGL setup, frame rendering, sprite sheet and its JSON.
// ---------- WebGL ----------
const glc=document.createElement('canvas');
const gl=glc.getContext('webgl2',{antialias:false,premultipliedAlpha:false,preserveDrawingBuffer:true});
const errEl=document.getElementById('err');
function fail(m){errEl.textContent=m;errEl.hidden=false;}
function shd(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
function prog(fs){const p=gl.createProgram();gl.attachShader(p,shd(gl.VERTEX_SHADER,VS));gl.attachShader(p,shd(gl.FRAGMENT_SHADER,fs));gl.bindAttribLocation(p,0,'a');gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;}
const N1=['uOff','uFrame','uPhase','uYaw','uElev','uScale','uAnim','uPolish','uHead','uTilt','uJaw','uEyeL','uEyeR','uBoneR','uBoneX','uMissing','uWound','uHunch','uArmGone','uLimp','uHS','uBuild','uShoulder','uHair','uCloth','uSex','uShoe','uArmShadow'];
const N2=['uTex','uCell','uPal','uOut','uShadow','uOutline','uPolish'];
let P1,P2,U1={},U2={},fb,tex;
let FW=64,FH=80,Z;

function setup(){
  P1=prog(FS1);P2=prog(FS2);
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  N1.forEach(n=>U1[n]=gl.getUniformLocation(P1,n));N2.forEach(n=>U2[n]=gl.getUniformLocation(P2,n));
  fb=gl.createFramebuffer();tex=gl.createTexture();
}
const ROWS=ANIMS.length*DIRS.length;
const polishEl=document.getElementById('polish');
// Render a list of frames [animIndex, dirIndex, phase] for specimen z into the GL canvas, `cols` per row.
function drawFrames(z,list,cols){
  const rows=Math.ceil(list.length/cols),W=FW*cols,H=FH*rows,pol=polishEl.checked?1:0;
  glc.width=W;glc.height=H;
  gl.bindTexture(gl.TEXTURE_2D,tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,W,H,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.bindFramebuffer(gl.FRAMEBUFFER,fb);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
  gl.viewport(0,0,W,H);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(P1);
  const f=(n,v)=>gl.uniform1f(U1[n],v);
  gl.uniform2f(U1.uFrame,FW,FH);
  f('uElev',(+document.getElementById('elev').value)*Math.PI/180);f('uScale',1.38);
  f('uPolish',pol);f('uHead',z.head);
  f('uTilt',z.tilt);f('uJaw',z.jaw);f('uEyeL',z.eyeL);f('uEyeR',z.eyeR);f('uBoneR',z.boneR);f('uBoneX',z.boneX);
  f('uMissing',z.missing);f('uWound',z.wound);f('uHunch',z.hunch);f('uArmGone',z.armGone);f('uLimp',z.limp);
  f('uHS',z.height);f('uBuild',z.build);f('uShoulder',z.shoulder);f('uHair',z.hair);f('uCloth',z.cloth);f('uSex',z.sex);f('uShoe',z.shoe);f('uArmShadow',window.ARM_SHADOW?1:0);
  list.forEach(([ai,di,ph],i)=>{
    const x=(i%cols)*FW,y=(rows-1-Math.floor(i/cols))*FH;
    gl.viewport(x,y,FW,FH);gl.uniform2f(U1.uOff,x,y);
    f('uAnim',ai);f('uYaw',DIRS[di].yaw);f('uPhase',ph);
    gl.drawArrays(gl.TRIANGLES,0,3);
  });
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,W,H);
  gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(P2);
  gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tex);gl.uniform1i(U2.uTex,0);
  gl.uniform2f(U2.uCell,FW,FH);gl.uniform3fv(U2.uPal,paletteArray(z));
  gl.uniform3f(U2.uOut,14/255,10/255,20/255);gl.uniform3f(U2.uShadow,10/255,8/255,16/255);
  gl.uniform1i(U2.uOutline,document.getElementById('outline').checked?1:0);
  gl.uniform1i(U2.uPolish,pol);
  gl.drawArrays(gl.TRIANGLES,0,3);
  return {W,H};
}
function renderSheet(){
  const list=[];
  ANIMS.forEach((an,ai)=>DIRS.forEach((d,di)=>{
    // looping clips wrap around; one-shot clips run from first pose to last
    for(let k=0;k<FRAMES;k++)list.push([ai,di,an.loop?k/FRAMES:k/(FRAMES-1)]);
  }));
  const {W,H}=drawFrames(Z,list,FRAMES);
  const url=glc.toDataURL('image/png');
  const img=document.getElementById('sheetImg');
  img.src=url;img.style.width=(W*2)+'px';
  const bm=new Image();bm.onload=()=>{sheetBitmap=bm;};bm.src=url;
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

