// SDF Walkers v0.16.0 · GLSL sources: pass 1 raymarches the SDF zombie, pass 2 turns light/material/depth into palette pixels and outlines.
const VS=`#version 300 es
in vec2 a;void main(){gl_Position=vec4(a,0,1);}`;

const FS1=`#version 300 es
precision highp float;
uniform vec2 uOff,uFrame;
uniform float uPhase,uYaw,uElev,uScale,uAnim,uPolish,uHead;
uniform float uTilt,uJaw,uEyeL,uEyeR,uBoneR,uBoneX,uMissing,uWound;
uniform float uHunch,uArmGone,uLimp,uHS,uBuild,uShoulder,uHair,uCloth,uSex,uShoe,uArmShadow;
out vec4 o;
// pose, computed once per pixel in main()
float gSw[2],gBend[2],gArmA[2],gArmB[2];
float gHipY,gBob,gHunch,gReach,gJaw,gHeadPitch,gSwayZ,gEyeOff;
bool gNoArms=false;   // set while tracing shadows: arms held forward would stripe the chest

float sph(vec3 p,vec3 c,float r){return length(p-c)-r;}
float ell(vec3 p,vec3 c,vec3 r){vec3 q=(p-c)/r;float k0=length(q),k1=length(q/r);return k0*(k0-1.)/max(k1,1e-6);}
float box(vec3 p,vec3 c,vec3 b,float r){vec3 q=abs(p-c)-b+r;return length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.)-r;}
float cap(vec3 p,vec3 a,vec3 b,float r){vec3 pa=p-a,ba=b-a;float h=clamp(dot(pa,ba)/dot(ba,ba),0.,1.);return length(pa-ba*h)-r;}
float smin(float a,float b,float k){float h=clamp(.5+.5*(b-a)/k,0.,1.);return mix(b,a,h)-k*h*(1.-h);}
float smax(float a,float b,float k){return -smin(-a,-b,k);}
vec3 rotX(vec3 p,float a){float c=cos(a),s=sin(a);return vec3(p.x,c*p.y-s*p.z,s*p.y+c*p.z);}
vec3 rotY(vec3 p,float a){float c=cos(a),s=sin(a);return vec3(c*p.x-s*p.z,p.y,s*p.x+c*p.z);}
void U(inout vec2 r,float d,float m){if(d<r.x)r=vec2(d,m);}

void pose(){
  float ph=uPhase,t=ph*6.2831853;
  float fv=abs(cos(uYaw))*uPolish;   // 1 in front/back views when polish is on
  gHunch=uHunch;gJaw=uJaw;gHeadPitch=0.;gSwayZ=0.;gBob=0.;gReach=0.;gHipY=1.02;gEyeOff=0.;
  for(int i=0;i<2;i++){gSw[i]=0.;gBend[i]=.05;gArmA[i]=1.45;gArmB[i]=-.25;}
  if(uAnim<.5){ // walk
    gBob=.035*cos(2.*t)*(1.+fv);
    gSwayZ=.08*fv*sin(t);
    for(int i=0;i<2;i++){
      float p2=t+(i==0?0.:3.14159);
      float lim=i==1?1.-uLimp:1.;
      gSw[i]=.42*lim*sin(p2);
      gBend[i]=.85*max(0.,cos(p2))*(i==1?1.-uLimp*.7:1.)*(1.+.9*fv);
      // arms are not a matched pair: one rides higher, wrists droop, both drift with the step
      gArmA[i]=1.45+(i==0?-.12:.1)+(.12+.15*fv)*sin(t+float(i)*1.7);
      gArmB[i]=-.3-.1*sin(t+float(i)*1.3);
    }
  }else if(uAnim<1.5){ // idle
    gHipY=1.01-.012*sin(t);
    gSwayZ=.05*sin(t);
    gHunch+=.04*sin(t);
    gHeadPitch=.06*sin(t+.8);
    gJaw+=.025*max(0.,sin(2.*t));
    for(int i=0;i<2;i++){gArmA[i]=1.05+(i==0?-.08:.08)+.1*sin(t+float(i)*2.);gArmB[i]=.3;}
  }else if(uAnim<2.5){ // attack
    float w=smoothstep(0.,.35,ph)*(1.-smoothstep(.35,.5,ph));
    float l=smoothstep(.35,.5,ph)*(1.-smoothstep(.75,1.,ph));
    gSw[0]=.25;gSw[1]=-.2;gBend[0]=.15;gBend[1]=.1;gHipY=.99-.04*l;
    gHunch+=-.2*w+.2*l;
    gHeadPitch=-.25*w+.25*l;
    gJaw+=.08*l;
    gReach=.1*l;
    for(int i=0;i<2;i++){gArmA[i]=1.4+1.*w-.15*l+(i==0?-.06:.06);gArmB[i]=-.15+.85*w;}
  }else if(uAnim<3.5){ // hit
    float k=ph<.2?ph/.2:1.-smoothstep(.2,1.,ph);
    gHunch-=.55*k;
    gHeadPitch=-.6*k;
    gJaw+=.05*k;
    gHipY=1.02-.03*k;
    gSw[0]=-.15*k;gSw[1]=.12*k;
    for(int i=0;i<2;i++){gArmA[i]=1.45+(i==0?.9:.55)*k;gArmB[i]=-.25+.85*k;}
  }else{ // death: knees buckle, then the body slumps forward
    float kn=smoothstep(0.,.45,ph),sl=smoothstep(.35,.9,ph);
    for(int i=0;i<2;i++){
      gSw[i]=.9*kn*(i==0?1.:.85);gBend[i]=2.2*kn;
      gArmA[i]=mix(1.45,.25,kn);gArmB[i]=.2;
    }
    float a=gSw[0],b=gSw[0]-gBend[0];
    gHipY=.07+max(.48*cos(a)+.46*cos(b),.48*cos(a)-.02);
    gHunch+=.8*sl;
    gHeadPitch=.45*sl;
    gJaw+=.05*kn;
    gEyeOff=step(.8,ph);
  }
}

vec2 headMap(vec3 p){
  float c=cos(uTilt),s=sin(uTilt);
  p.xy=vec2(c*p.x-s*p.y,s*p.x+c*p.y);
  vec3 q=vec3(abs(p.x),p.yz);
  float d=ell(p,vec3(0,.2,-.05),vec3(.7,.75,.75));
  d=smin(d,ell(p,vec3(0,-.35,.12),vec3(.5,.45,.55)),.25);
  d=smin(d,ell(q,vec3(.38,-.05,.45),vec3(.2,.15,.2)),.12);
  d=smin(d,cap(q,vec3(.08,.38,.62),vec3(.45,.42,.5),.09),.08);
  d=smax(d,-sph(q,vec3(.5,-.3,.5),.17),.1);
  d=smin(d,cap(p,vec3(0,.25,.75),vec3(0,-.02,.86),.07),.06);
  d=smin(d,ell(p,vec3(.7,.05,-.05),vec3(.1,.2,.1)),.05);
  d=smin(d,ell(p,vec3(-.7,-.02,-.05),vec3(.1,.12,.1)),.05);
  float sock=smin(sph(p,vec3(-.27,.2,.62),.16),ell(p,vec3(.27,.15,.62),vec3(.17,.12,.16)),.01);
  d=smax(d,-sock,.04);
  float eyes=1e9;
  if(uEyeL>0.)eyes=min(eyes,sph(p,vec3(-.27,.19,.55+.08*uPolish),uEyeL*(1.4+1.3*uPolish)));
  if(uEyeR>0.)eyes=min(eyes,sph(p,vec3(.29,.13,.55+.08*uPolish),uEyeR*(1.4+1.3*uPolish)));
  d=smax(d,-ell(p,vec3(.04,-.42,.62),vec3(.28,gJaw,.25)),.04);
  float mouth=ell(p,vec3(.04,-.42,.45),vec3(.27,gJaw-.01,.1));
  float top=-.42+gJaw,teeth=1e9;
  for(int i=0;i<5;i++){
    if(float(i)==uMissing)continue;
    float x=-.18+.095*float(i),h=.05+.02*float((i*7)%3);
    teeth=min(teeth,box(p,vec3(x,top-.04-h*.5,.7),vec3(.035,h*.5,.04),.01));
  }
  float br=uBoneR>.01?sph(p,vec3(uBoneX,.62,.4),uBoneR)+.06*sin(14.*p.x)*sin(11.*p.z+1.3):1e9;
  float skin=uBoneR>.01?smax(d,-br,.02):d;
  float bone=uBoneR>.01?smax(d+.04,br,.02):1e9;
  float wound=cap(p,vec3(.62*uWound,.2,.3),vec3(.45*uWound,-.3,.55),.09);
  // hair: 0 bald, 1 short, 2 long strands, 3 patchy
  float hair=1e9;
  if(uHair>.5){
    float hc=ell(p,vec3(0,.25,-.1),vec3(.76,.8,.8));
    hc=smax(hc,dot(p,normalize(vec3(0,-.5,.87)))-.18,.04);   // hairline keeps the face clear
    hc=smax(hc,-.05-p.y,.04);                                  // stops above the ears
    if(uHair>2.5)hc=smax(hc,-min(sph(p,vec3(.5,.6,.15),.42),min(sph(p,vec3(-.15,.85,-.25),.38),sph(p,vec3(-.6,.25,-.4),.36))),.03);
    hair=hc;
    if(uHair>1.5&&uHair<2.5){
      for(int i=0;i<3;i++){
        float a=-1.15+1.15*float(i);
        vec3 s0=vec3(.6*sin(a),.2,-.6*cos(a)-.05);
        hair=smin(hair,cap(p,s0,vec3(s0.x*1.15,-1.15+.15*float(i%2),s0.z*1.1-.1),.22),.12);
      }
      hair=smax(hair,dot(p,normalize(vec3(0,-.25,.97)))-.3,.05);   // keep the strands off the face
    }
    if(uBoneR>.01)hair=smax(hair,-br,.02);
  }
  vec2 r=vec2(skin,0.);
  U(r,hair,8.);
  U(r,eyes,gEyeOff>.5?2.:1.);U(r,mouth,2.);U(r,teeth,3.);U(r,bone,4.);
  if(r.y==0.&&wound<.02)r.y=5.;
  return r;
}

vec2 map(vec3 p){
  p=rotY(p,-uYaw);
  float cz=cos(-gSwayZ),sz=sin(-gSwayZ);
  p.xy=vec2(cz*p.x-sz*p.y,sz*p.x+cz*p.y);
  p.y-=gBob;
  p/=uHS;                                   // height scales the whole figure about the feet
  vec2 r=vec2(1e9,0.);
  float dy=gHipY-1.02,lt=1.+.15*uPolish;
  float bw=.85+.4*uBuild,lb=.9+.3*uBuild;   // girth and limb thickness
  bool fem=uSex>.5;
  // garments: 0 shirt, 1 jacket, 2 hospital gown, 3 sleeveless, 4 overalls, 5 rags, 6 shirtless (crop top on women), 7 dress, 8 shorts
  int g=int(uCloth+.5);
  bool jacket=g==1,gown=g==2,overalls=g==4,rags=g==5,bare=g==6,dress=g==7,shorts=g==8;
  bool bareArms=gown||g==3||bare||dress,skirt=gown||dress;
  // legs
  for(int i=0;i<2;i++){
    float s=i==0?-1.:1.;
    vec3 hip=vec3(.135*(.92+.16*uBuild)*(fem?1.18:1.)*s,gHipY,0);
    vec3 knee=hip+vec3(0,-.48*cos(gSw[i]),.48*sin(gSw[i]));
    float a2=gSw[i]-gBend[i];
    vec3 ank=knee+vec3(0,-.46*cos(a2),.46*sin(a2));
    U(r,cap(p,hip,knee,.085*lt*lb),skirt?0.:7.);
    U(r,cap(p,knee,ank,.065*lt*lb),skirt||shorts?0.:7.);
    float fs=uShoe>.5&&uShoe<1.5?1.12:uShoe>2.5&&uShoe<3.5?.92:1.;   // sneakers chunky, dress shoes slim
    U(r,ell(p,ank+rotX(vec3(.015*s,-.03,.07),-a2*.5),vec3(.065*fs,.05,.13*fs)*lt),uShoe<.5?0.:9.);
    if(uShoe>1.5&&uShoe<2.5||uShoe>3.5){
      float bh=uShoe>3.5?.55:.38;                        // boot shaft up the shin
      U(r,cap(p,ank,ank+(knee-ank)*bh,.075*lt*lb),9.);
    }
  }
  U(r,ell(p,vec3(0,1.04+dy,0),vec3(.23*bw*(fem?1.28:1.),.13,.14*bw)),skirt?6.:7.);
  // hunched torso
  vec3 base=vec3(0,1.07+dy,0);
  vec3 q=rotX(p-base,-gHunch);
  float chestX=.27*(.88+.2*uShoulder)*(.92+.16*uBuild)*(fem?.85:1.);
  float tor=smin(ell(q,vec3(0,.47,0),vec3(chestX,.3,.16*bw)),ell(q,vec3(0,.17,.02*uBuild),vec3(.21*bw*(fem?.82:1.),.22,.14*bw*(1.+.3*uBuild))),.1);
  if(fem)tor=smin(tor,min(ell(q,vec3(-.085,.5,.12*bw),vec3(.085,.075,.07)),ell(q,vec3(.085,.5,.12*bw),vec3(.085,.075,.07))),.05);
  if(jacket)tor-=.014;
  tor=smax(tor,-sph(q,vec3(.18,-.02,.16),.07),.02);
  bool chest=q.y>.36&&q.y<.64;             // the band a top always covers on women
  float tm=6.;
  if(overalls&&(q.y<.52||(abs(abs(q.x)-.12)<.045&&q.z>0.)))tm=7.;
  if(bare)tm=fem&&chest?6.:0.;
  if(rags){
    float hx=fract(uBoneX*7.31+.5)*2.-1.;     // per-zombie placement of the tears
    bool tear=q.y<.07+.06*sin(q.x*21.+hx*4.)             // ragged hem
      ||length(q-vec3(.13*hx,.24,.15))<.075
      ||length(q-vec3(-.12*hx,.45,.14))<.06
      ||length(q-vec3(.06*hx,.33,-.15))<.085;           // one on the back
    if(tear&&!(fem&&chest))tm=0.;
  }
  U(r,tor,tm);
  if(gown)U(r,ell(q,vec3(0,-.13,0),vec3(.25*bw,.37,.18*bw)),6.);
  if(dress)U(r,smin(ell(q,vec3(0,-.04,0),vec3(.25*bw,.26,.18*bw)),ell(q,vec3(0,-.3,0),vec3(.33*bw,.1,.25*bw)),.12),6.);
  float hs=uHead;
  vec3 nb=rotX(vec3(0,.74,0),gHunch)+base;
  vec3 hc=rotX(vec3(0,.8+hs*.9,.05),gHunch*1.2)+base+vec3(0,0,.03);
  U(r,cap(p,nb,hc-vec3(0,hs*.8,.01),.055*(.9+.3*uShoulder)*(fem?.9:1.)),0.);
  if(jacket)U(r,cap(p,nb+vec3(-.09,-.03,-.02),nb+vec3(.09,-.03,-.02),.055),6.);
  // arms: angle 0 hangs down, about 1.5 points forward, above 2 is raised
  float sx=(.25+.06*uShoulder+.02*uBuild)*(fem?.85:1.),ar=lb*(.9+.25*uShoulder)*(fem?.9:1.);
  for(int i=0;i<2;i++){
    float s=i==0?-1.:1.;
    if(gNoArms)break;
    vec3 sh=rotX(vec3(sx*s,.66,0),gHunch)+base+vec3(0,0,gReach);
    if(uArmGone==s){
      U(r,cap(p,sh,sh+vec3(.03*s,-.13,.05),.09*ar),bareArms?0.:6.);
      U(r,sph(p,sh+vec3(.03*s,-.16,.06),.06*ar),5.);
      continue;
    }
    float A=gArmA[i],B=A+gArmB[i];
    vec3 el=sh+vec3(-.02*s,-.34*cos(A),.34*sin(A));
    vec3 hd=el+vec3(-.03*s,-.3*cos(B),.3*sin(B));
    U(r,cap(p,sh,el,.065*ar),bareArms?0.:6.);
    U(r,cap(p,el,hd,.05*ar),jacket?6.:0.);
    U(r,sph(p,hd+.04*normalize(hd-el)+vec3(0,-.02,0),.055),0.);
  }
  // head
  vec3 hl=rotX((p-hc)/hs,-.18-gHeadPitch);
  vec2 h=headMap(hl);h.x*=hs;
  U(r,h.x,h.y);
  r.x*=uHS;
  return r;
}

void main(){
  pose();
  vec2 uv=((gl_FragCoord.xy-uOff)/uFrame*2.-1.)*vec2(uFrame.x/uFrame.y,1.)*uScale;
  float el=uElev;
  vec3 fw=normalize(vec3(0,-sin(el),-cos(el)));
  vec3 rt=vec3(1,0,0),up=cross(rt,fw);
  vec3 ta=vec3(0,1.2,0)+rotY(vec3(0,0,.22),uYaw);
  vec3 ro=ta+uv.x*rt+uv.y*up-fw*6.,rd=fw;
  float t=0.;bool hit=false;
  for(int i=0;i<120;i++){float d=map(ro+rd*t).x;if(d<1e-3){hit=true;break;}t+=d*.9;if(t>12.)break;}
  if(!hit){
    if(rd.y<-.05){
      vec3 g=ro+rd*(-ro.y/rd.y);
      if(g.x*g.x/.1+g.z*g.z/.07<1.){o=vec4(0,0,0,.5);return;}
    }
    o=vec4(0);return;
  }
  vec3 p=ro+rd*t;vec2 e=vec2(1e-3,0);
  vec3 n=normalize(vec3(map(p+e.xyy).x-map(p-e.xyy).x,map(p+e.yxy).x-map(p-e.yxy).x,map(p+e.yyx).x-map(p-e.yyx).x));
  vec3 L=normalize(vec3(.6,.8,.6));
  float dif=clamp(dot(n,L),0.,1.),sh=1.,ts=.02;
  gNoArms=uArmShadow<.5;
  for(int i=0;i<24;i++){float h=map(p+n*2e-3+L*ts).x;sh=min(sh,clamp(8.*h/ts,0.,1.));ts+=clamp(h,.01,.2);}
  gNoArms=false;
  float ao=1.;
  for(int i=1;i<5;i++){float h=.04*float(i);ao-=(h-map(p+n*h).x)*pow(.5,float(i))*4.;}
  ao=clamp(ao,0.,1.);
  float rim=pow(clamp(1.+dot(n,rd),0.,1.),3.);
  float l=.15+.85*dif*sh*(.5+.5*ao)+.25*rim;
  o=vec4(clamp(l,0.,1.),map(p).y/16.,t/12.,1.);   // continuous light; the palette pass quantizes it
}`;

const FS2=`#version 300 es
precision highp float;
uniform sampler2D uTex;uniform vec2 uCell;uniform vec3 uPal[40];uniform vec3 uOut,uShadow;uniform int uOutline,uPolish;
out vec4 o;
ivec2 lo,hi;
const ivec2 N[4]=ivec2[4](ivec2(1,0),ivec2(-1,0),ivec2(0,1),ivec2(0,-1));
vec4 T(ivec2 c){return texelFetch(uTex,clamp(c,lo,hi),0);}
int bandOf(float l){return l<.18?0:l<.45?1:l<.75?2:3;}
float bayer(ivec2 c){int i=(c.x&1)+2*(c.y&1);return i==0?.125:i==1?.625:i==2?.875:.375;}
// palette color of a pixel; x=-1 when it is empty or shadow
vec3 colAt(ivec2 c,bool dither,out int m,out int b){
  vec4 s=T(c);m=-1;b=0;
  if(s.a<.75)return vec3(-1);
  m=int(round(s.g*16.));
  float l=s.r;
  if(dither&&(m==6||m==7))l+=(bayer(c)-.5)*.07;   // only pixels right at a band edge flip
  b=m==1?3:bandOf(l);
  return uPal[m*4+b];
}
bool same(vec3 a,vec3 b){return distance(a,b)<1e-3;}
void main(){
  ivec2 c=ivec2(gl_FragCoord.xy),cell=ivec2(uCell);
  lo=(c/cell)*cell;hi=lo+cell-1;
  vec4 s=T(c);
  if(s.a<.25){o=vec4(0);return;}
  if(s.a<.75){o=vec4(uShadow,.45);return;}
  int m,b,mm,bb;
  vec3 base=colAt(c,false,m,b);
  vec3 col=base;
  if(uPolish==1){
    col=colAt(c,true,m,b);
    // stray-pixel cleanup on the undithered image; keep tiny features (eyes, teeth, wounds)
    if(m==0||m==4||m==6||m==7){
      vec3 n0=colAt(c+N[0],false,mm,bb),n1=colAt(c+N[1],false,mm,bb),n2=colAt(c+N[2],false,mm,bb),n3=colAt(c+N[3],false,mm,bb);
      if(!same(base,n0)&&!same(base,n1)&&!same(base,n2)&&!same(base,n3)){
        if(same(n0,n1)&&n0.x>=0.)col=n0;
        else if(same(n2,n3)&&n2.x>=0.)col=n2;
      }
    }
  }
  if(uOutline==1){
    bool edge=false,silh=false;
    for(int i=0;i<4;i++){
      vec4 n=T(c+N[i]);
      if(n.a<.75){edge=true;silh=true;}
      else if(n.b-s.b>.2/12.)edge=true;
      if(i%2==0&&n.a>.75&&int(round(n.g*16.))!=m&&m!=1)edge=true;
    }
    if(edge){
      // selective outline: the lit side of the silhouette uses the material's own darkest tone
      if(uPolish==1&&silh&&b>=2&&m!=1)col=mix(uPal[m*4],uOut,.35);
      else col=uOut;
    }
  }
  o=vec4(col,1);
}`;

