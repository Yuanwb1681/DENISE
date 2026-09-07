#include <cuda_runtime.h>
#include <stdio.h>
#include "denise_gpu.h"

__global__ static void store_psv_k(float *vx,float *vy,float *sxx,float *syy,float *sxy,
 float *ux,float *uy,float *uxy,float *frx,float *fry,float *fx,float *fy,float *fu,
 int st,int nx,int ny,int idx,int idy,int off,int grad){
 int q=blockIdx.x*blockDim.x+threadIdx.x,nyi=(ny+idy-1)/idy,nxi=(nx+idx-1)/idx,n=nxi*nyi;
 if(q>=n)return;int x=(q/nyi)*idx,y=(q%nyi)*idy,k=y*st+x,z=off+q;
 frx[z]=vx[k];fry[z]=vy[k];fx[z]=grad==1?sxx[k]:ux[k];fy[z]=grad==1?syy[k]:uy[k];fu[z]=grad==1?sxy[k]:uxy[k];
}
__global__ static void corr_psv_k(float *vx,float *vy,float *sxx,float *syy,float *sxy,
 float *gl,float *gm,float *gr,const float *frx,const float *fry,const float *fx,
 const float *fy,const float *fu,const float *rho,const float *vp,const float *vs,
 int st,int nx,int ny,int idx,int idy,int off,int grad,int invmat){
 int q=blockIdx.x*blockDim.x+threadIdx.x,nyi=(ny+idy-1)/idy,nxi=(nx+idx-1)/idx,n=nxi*nyi;
 if(q>=n)return;int x=(q/nyi)*idx,y=(q%nyi)*idy,k=y*st+x,z=off+q;
 gr[k]+=vx[k]*frx[z]+vy[k]*fry[z];float fsum=fx[z]+fy[z],ssum=sxx[k]+syy[k];gl[k]+=fsum*ssum;
 float mu,lam;if(invmat==1){mu=rho[k]*vs[k]*vs[k];lam=rho[k]*vp[k]*vp[k]-2.0f*mu;}else{mu=vs[k];lam=vp[k];}
 if(mu<=0.0f)return;
 if(grad==1){gm[k]+=fu[z]*sxy[k]/(mu*mu)+0.25f*fsum*ssum/((lam+mu)*(lam+mu))+0.25f*(fx[z]-fy[z])*(sxx[k]-syy[k])/(mu*mu);}
 else{float a=0.25f/((lam+mu)*(lam+mu)),b=0.25f/(mu*mu);gm[k]+=fu[z]*sxy[k]/(mu*mu)+(a+b)*(fx[z]*sxx[k]+fy[z]*syy[k])+(a-b)*(fx[z]*syy[k]+fy[z]*sxx[k]);}
}
__global__ static void corr_aniso_k(float *vx,float *vy,float *sxx,float *syy,
 float *gl,float *gr,const float *frx,const float *fry,const float *fx,const float *fy,
 int st,int nx,int ny,int idx,int idy,int off){int q=blockIdx.x*blockDim.x+threadIdx.x,
 nyi=(ny+idy-1)/idy,nxi=(nx+idx-1)/idx,n=nxi*nyi;if(q>=n)return;
 int x=(q/nyi)*idx,y=(q%nyi)*idy,k=y*st+x,z=off+q;
 gr[k]+=vx[k]*frx[z]+vy[k]*fry[z];gl[k]+=(fx[z]+fy[z])*(sxx[k]+syy[k]);}

__global__ static void store_ac_k(float *vx,float *vy,float *p,float *ux,float *frx,
 float *fry,float *fp,int st,int nx,int ny,int idx,int idy,int off,int grad){
 int q=blockIdx.x*blockDim.x+threadIdx.x,nyi=(ny+idy-1)/idy,nxi=(nx+idx-1)/idx,n=nxi*nyi;
 if(q>=n)return;int x=(q/nyi)*idx,y=(q%nyi)*idy,k=y*st+x,z=off+q;
 frx[z]=vx[k];fry[z]=vy[k];fp[z]=(grad==1?p[k]:ux[k]);
}
__global__ static void corr_ac_k(float *vx,float *vy,float *p,float *gl,float *gr,
 const float *frx,const float *fry,const float *fp,int st,int nx,int ny,int idx,int idy,int off){
 int q=blockIdx.x*blockDim.x+threadIdx.x,nyi=(ny+idy-1)/idy,nxi=(nx+idx-1)/idx,n=nxi*nyi;
 if(q>=n)return;int x=(q/nyi)*idx,y=(q%nyi)*idy,k=y*st+x,z=off+q;
 gl[k]+=fp[z]*p[k];gr[k]+=vx[k]*frx[z]+vy[k]*fry[z];
}
__global__ static void store_sh_k(float *vz,float *sxz,float *syz,float *uzx,float *uz,
 float *fr,float *fx,float *fy,int st,int nx,int ny,int idx,int idy,int off,int grad){
 int q=blockIdx.x*blockDim.x+threadIdx.x,nyi=(ny+idy-1)/idy,nxi=(nx+idx-1)/idx,n=nxi*nyi;
 if(q>=n)return;int x=(q/nyi)*idx,y=(q%nyi)*idy,k=y*st+x,z=off+q;
 fr[z]=vz[k];fx[z]=(grad==1?sxz[k]:uzx[k]);fy[z]=(grad==1?syz[k]:uz[k]);
}
__global__ static void corr_sh_k(float *vz,float *sxz,float *syz,float *gr,float *gu,
 const float *fr,const float *fx,const float *fy,const float *rho,const float *mu,
 int st,int nx,int ny,int idx,int idy,int off,int invmat){
 int q=blockIdx.x*blockDim.x+threadIdx.x,nyi=(ny+idy-1)/idy,nxi=(nx+idx-1)/idx,n=nxi*nyi;
 if(q>=n)return;int x=(q/nyi)*idx,y=(q%nyi)*idy,k=y*st+x,z=off+q;
 gr[k]+=vz[k]*fr[z];float m=invmat==1?rho[k]*mu[k]*mu[k]:mu[k];
 if(m>0.0f)gu[k]+=fy[z]*syz[k]+fx[z]*sxz[k];
}
static int done(const char *where){cudaError_t e=cudaPeekAtLastError();if(e!=cudaSuccess){fprintf(stderr,"%s launch: %s\n",where,cudaGetErrorString(e));return -1;}return 0;}
__global__ static void eprecond_k(float *w,const float *vx,const float *vy,
 int stride,int nx,int ny,int idx,int idy)
{
 int q=blockIdx.x*blockDim.x+threadIdx.x,nxq=(nx+idx-1)/idx,nyq=(ny+idy-1)/idy;
 if(q>=nxq*nyq)return;int x=(q/nyq)*idx,y=(q%nyq)*idy,k=y*stride+x;
 float xvel=vx[k],yvel=vy[k];w[k]+=xvel*xvel+yvel*yvel;
}
extern "C" int denise_gpu_eprecond(float **w,float **vx,float **vy,
 int nx,int ny,int idx,int idy)
{
 int stride=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);
 eprecond_k<<<(n+255)/256,256>>>(w[1]+1,vx[1]+1,vy[1]+1,stride,nx,ny,idx,idy);
 return done("FWI energy precondition CUDA");
}

__device__ static float positive_atomic_max(float *address,float value)
{
 unsigned int *p=(unsigned int *)address,old=*p,assumed;
 while(__uint_as_float(old)<value){assumed=old;old=atomicCAS(p,assumed,__float_as_uint(value));if(old==assumed)break;}
 return __uint_as_float(old);
}

__global__ static void finish_prepare_psv(float *gvp,float *gvs,float *grho,
 float *glam,float *gmu,float *grhos,float *ws,float *we,const float *vp,
 const float *vs,const float *rho,int stride,int nx,int ny,int idx,int idy,
 int invmat,int grad,int iter,int ivp,int ivs,int irho,float dt,float dh,
 int global_nx,float *maxwe)
{
 int q=blockIdx.x*blockDim.x+threadIdx.x,nxq=(nx+idx-1)/idx,nyq=(ny+idy-1)/idy;
 if(q>=nxq*nyq)return;int x=(q/nyq)*idx,y=(q%nyq)*idy,k=y*stride+x;
 float lam=-dt*gvp[k],mu=-dt*gvs[k],muss,lamss;
 if(invmat==1){muss=rho[k]*vs[k]*vs[k];lamss=rho[k]*vp[k]*vp[k]-2.0f*muss;
  if(muss>0.0f||lamss>0.0f)lam/=4.0f*(lamss+muss)*(lamss+muss);
  gvp[k]=2.0f*vp[k]*rho[k]*lam;gvs[k]=-4.0f*rho[k]*vs[k]*lam+2.0f*rho[k]*vs[k]*mu;
  grhos[k]=-dt*grho[k];grho[k]=(vp[k]*vp[k]-2.0f*vs[k]*vs[k])*lam+vs[k]*vs[k]*mu+grhos[k];
 }else if(invmat==3){muss=vs[k];lamss=vp[k];if(grad==1&&(muss>0.0f||lamss>0.0f))lam/=4.0f*(lamss+muss)*(lamss+muss);gvp[k]=lam;gvs[k]=mu;grhos[k]=-dt*grho[k];grho[k]=grhos[k];
 }else{gvp[k]=2.0f*vp[k]*lam;gvs[k]=-4.0f*vs[k]*lam+2.0f*vs[k]*mu;}
 glam[k]=lam;gmu[k]=mu;if(iter<ivp)gvp[k]=0;if(iter<ivs)gvs[k]=0;if(iter<irho)grho[k]=0;
 float xx=(float)(x+1),yy=(float)(y+1),xmin=dh,xmax=(float)global_nx*dh;
 float weight=sqrtf(ws[k])*(asinhf((xmax-xx*dh)/(yy*dh))-asinhf((xmin-xx*dh)/(yy*dh)));
 we[k]=weight;positive_atomic_max(maxwe,weight);
}

__global__ static void finish_scale_stack_psv(float *gvp,float *gvs,float *grho,
 float *sumvp,float *sumvs,float *sumrho,float *we,int stride,int nx,int ny,
 int idx,int idy,float cvp,float cvs,float crho,const float *maxwe)
{
 int q=blockIdx.x*blockDim.x+threadIdx.x,nxq=(nx+idx-1)/idx,nyq=(ny+idy-1)/idy;
 if(q>=nxq*nyq)return;int x=(q/nyq)*idx,y=(q%nyq)*idy,k=y*stride+x;
 float w=we[k]+0.005f*(*maxwe);we[k]=w;gvp[k]/=w*cvp*cvp;
 if(cvs==0.0f)gvs[k]=0.0f;else gvs[k]/=w*cvs*cvs;grho[k]/=w*crho*crho;
 sumvp[k]+=gvp[k];sumvs[k]+=gvs[k];sumrho[k]+=grho[k];
}

extern "C" int denise_gpu_finish_shot_psv(float **gvp,float **gvs,float **grho,
 float **glam,float **gmu,float **grhos,float **sumvp,float **sumvs,float **sumrho,
 float **ws,float **we,float **vp,float **vs,float **rho,int nx,int ny,int idx,
 int idy,int invmat,int grad,int iter,int ivp,int ivs,int irho,float dt,float cvp,
 float cvs,float crho,float dh,int global_nx)
{
 int stride=gvp[2]-gvp[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);float *mx=NULL;
 cudaError_t e=cudaMalloc(&mx,sizeof(float));if(e!=cudaSuccess)return -1;cudaMemsetAsync(mx,0,sizeof(float));
 finish_prepare_psv<<<(n+255)/256,256>>>(gvp[1]+1,gvs[1]+1,grho[1]+1,glam[1]+1,
  gmu[1]+1,grhos[1]+1,ws[1]+1,we[1]+1,vp[1]+1,vs[1]+1,rho[1]+1,stride,
  nx,ny,idx,idy,invmat,grad,iter,ivp,ivs,irho,dt,dh,global_nx,mx);
 finish_scale_stack_psv<<<(n+255)/256,256>>>(gvp[1]+1,gvs[1]+1,grho[1]+1,
  sumvp[1]+1,sumvs[1]+1,sumrho[1]+1,we[1]+1,stride,nx,ny,idx,idy,cvp,cvs,crho,mx);
 e=cudaDeviceSynchronize();cudaFree(mx);if(e!=cudaSuccess){fprintf(stderr,"FWI finish shot CUDA: %s\n",cudaGetErrorString(e));return -1;}return 0;
}

__global__ static void residual_l2_k(float *obs,float *syn,float *res,
 int ostride,int sstride,int rstride,int ntr,int ns,int grad,int mode,
 int calc_l2,float dt,double *l2)
{
 int tr=blockIdx.x*blockDim.x+threadIdx.x;if(tr>=ntr)return;float integral=0.0f;
 double local=0.0;for(int j=1;j<=ns;j++){
  float s=syn[tr*sstride+j],o=obs[tr*ostride+j];if(j==1){o=s;obs[tr*ostride+j]=s;}
  float v;if(grad==1){if(mode==1||mode==3)integral+=dt*(s-o);else integral+=dt*o;v=integral;}
  else v=(mode==1||mode==3)?s-o:o;
  res[tr*rstride+(ns-j+1)]=v;if(calc_l2)local+=0.5*(double)v*(double)v;
 }
 if(calc_l2)atomicAdd(l2,local);
}

extern "C" int denise_gpu_residual_l2(float **obs,float **syn,float **res,
 int ntr,int ns,int grad,int mode,int calc_l2,float dt,double initial,double *result)
{
 if(ntr<=0){*result=initial;return 0;}int os=obs[2]-obs[1],ss=syn[2]-syn[1],rs=res[2]-res[1];
 double *dl2=NULL;cudaError_t e=cudaMalloc(&dl2,sizeof(double));if(e!=cudaSuccess)return -1;
 cudaMemcpy(dl2,&initial,sizeof(double),cudaMemcpyHostToDevice);
 residual_l2_k<<<(ntr+127)/128,128>>>(obs[1],syn[1],res[1],os,ss,rs,ntr,ns,
  grad,mode,calc_l2,dt,dl2);e=cudaDeviceSynchronize();
 if(e==cudaSuccess)e=cudaMemcpy(result,dl2,sizeof(double),cudaMemcpyDeviceToHost);cudaFree(dl2);
 if(e!=cudaSuccess){fprintf(stderr,"FWI L2 residual CUDA: %s\n",cudaGetErrorString(e));return -1;}return 0;
}

struct gpu_iir_coef { double f0,f1,f2,g1,g2; };
static gpu_iir_coef butter_coef(int high,double t0dt,double h)
{
 double eps=8.0*atan(1.0)/t0dt,epsq=eps*eps,a=1.0-eps*h+epsq/4.0;
 double b=-2.0+epsq/2.0,c=1.0+eps*h+epsq/4.0;gpu_iir_coef z;
 z.g1=-b/c;z.g2=-a/c;if(high){z.f0=1.0/c;z.f1=-2.0*z.f0;z.f2=z.f0;}
 else{z.f0=epsq/4.0/c;z.f1=2.0*z.f0;z.f2=z.f0;}return z;
}
__global__ static void filter_trace_k(float *data,int stride,int ntr,int ns,
 const gpu_iir_coef *coef,int stages,int high)
{
 int tr=blockIdx.x*blockDim.x+threadIdx.x;if(tr>=ntr)return;float *x=data+tr*stride;
 if(high){double x0=x[1];for(int j=1;j<=ns;j++)x[j]=(float)((double)x[j]-x0);}
 for(int s=0;s<stages;s++){double xa=0,xaa=0,ya=0,yaa=0;gpu_iir_coef c=coef[s];
  for(int j=1;j<=ns;j++){double xn=x[j],yn=c.f0*xn+c.f1*xa+c.f2*xaa+c.g1*ya+c.g2*yaa;
   x[j]=(float)yn;xaa=xa;xa=xn;yaa=ya;ya=yn;}}
}
extern "C" int denise_gpu_filter_traces(float **data,int ntr,int ns,int order,
 float fc,float dt,int high)
{
 if(ntr<=0||fc<=0||order<2)return -1;int stages=order/2,stride=data[2]-data[1];
 gpu_iir_coef host[32];if(stages>32)return -1;double pih=2.0*atan(1.0),t0dt=(1.0/(double)fc)/(double)dt;
 for(int j=1;j<=stages;j++)host[j-1]=butter_coef(high,t0dt,sin(pih*(2*j-1)/order));
 gpu_iir_coef *dev=NULL;cudaError_t e=cudaMalloc(&dev,stages*sizeof(*dev));if(e!=cudaSuccess)return -1;
 cudaMemcpy(dev,host,stages*sizeof(*dev),cudaMemcpyHostToDevice);
 filter_trace_k<<<(ntr+127)/128,128>>>(data[1],stride,ntr,ns,dev,stages,high);
 e=cudaDeviceSynchronize();cudaFree(dev);if(e!=cudaSuccess){fprintf(stderr,"FWI trace filter CUDA: %s\n",cudaGetErrorString(e));return -1;}return 0;
}
__global__ static void model_stats_psv_k(const float *gvp,const float *gvs,
 const float *grho,const float *vp,const float *vs,const float *rho,int stride,
 int nx,int ny,float *s){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=nx*ny)return;
 int x=q/ny,y=q%ny,k=y*stride+x;positive_atomic_max(s,vp[k]);
 positive_atomic_max(s+1,fabsf(gvp[k]));positive_atomic_max(s+2,vs[k]);
 positive_atomic_max(s+3,fabsf(gvs[k]));positive_atomic_max(s+4,rho[k]);
 positive_atomic_max(s+5,fabsf(grho[k]));}
extern "C" int denise_gpu_model_stats_psv(float **gvp,float **gvs,float **grho,
 float **vp,float **vs,float **rho,int nx,int ny,float s[6]){int st=gvp[2]-gvp[1],n=nx*ny;float *d=0;
 cudaError_t e=cudaMalloc(&d,6*sizeof(float));if(e!=cudaSuccess)return -1;cudaMemsetAsync(d,0,6*sizeof(float));
 model_stats_psv_k<<<(n+255)/256,256>>>(gvp[1]+1,gvs[1]+1,grho[1]+1,vp[1]+1,vs[1]+1,rho[1]+1,st,nx,ny,d);
 e=cudaDeviceSynchronize();if(e==cudaSuccess)e=cudaMemcpy(s,d,6*sizeof(float),cudaMemcpyDeviceToHost);cudaFree(d);
 if(e!=cudaSuccess){fprintf(stderr,"FWI model statistics CUDA: %s\n",cudaGetErrorString(e));return -1;}return 0;}
__global__ static void model_update_psv_k(const float *gvp,const float *gvs,const float *grho,
 float *vp,float *vs,float *rho,float *vpn,float *vsn,float *rhon,int st,int nx,int ny,
 int invmat,int commit,float evp,float evs,float erho,float vplo,float vphi,float vslo,
 float vshi,float rlo,float rhi){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=nx*ny)return;
 int x=q/ny,y=q%ny,k=y*st+x;float a=vp[k],b=vs[k],c=rho[k];if(invmat==1||invmat==3){
 float an=a-evp*gvp[k],bn=b-evs*gvs[k],cn=c-erho*grho[k];if(invmat==1){if(an<vplo||an>vphi)an=a;
 if((bn<vslo&&bn>1e-6f)||bn>vshi)bn=b;if(cn<rlo||cn>rhi)cn=c;}if(an<0)an=a;if(bn<0)bn=b;if(cn<0)cn=c;
 vpn[k]=an;vsn[k]=bn;rhon[k]=cn;if(commit){vp[k]=an;vs[k]=bn;rho[k]=cn;}}}
extern "C" int denise_gpu_model_update_psv(float **gvp,float **gvs,float **grho,float **vp,float **vs,float **rho,
 float **vpn,float **vsn,float **rhon,int nx,int ny,int invmat,int commit,float evp,float evs,float erho,
 float vplo,float vphi,float vslo,float vshi,float rlo,float rhi){int st=gvp[2]-gvp[1],n=nx*ny;
 model_update_psv_k<<<(n+255)/256,256>>>(gvp[1]+1,gvs[1]+1,grho[1]+1,vp[1]+1,vs[1]+1,rho[1]+1,
 vpn[1]+1,vsn[1]+1,rhon[1]+1,st,nx,ny,invmat,commit,evp,evs,erho,vplo,vphi,vslo,vshi,rlo,rhi);
 cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess){fprintf(stderr,"FWI model update CUDA: %s\n",cudaGetErrorString(e));return -1;}return 0;}
__global__ static void scale_copy_psv_k(float *vp,float *vs,float *rho,float *vpc,
 float *vsc,float *rhoc,int st,int nx,int ny,int idx,int idy,float avp,float avs,float arho){
 int q=blockIdx.x*blockDim.x+threadIdx.x,nxq=(nx+idx-1)/idx,nyq=(ny+idy-1)/idy;
 if(q>=nxq*nyq)return;int x=(q/nyq)*idx,y=(q%nyq)*idy,k=y*st+x;
 float a=vp[k]*avp,b=vs[k]*avs,c=rho[k]*arho;vp[k]=a;vs[k]=b;rho[k]=c;
 vpc[k]=a;vsc[k]=b;rhoc[k]=c;}
extern "C" int denise_gpu_scale_copy_psv(float **vp,float **vs,float **rho,float **vpc,
 float **vsc,float **rhoc,int nx,int ny,int idx,int idy,float avp,float avs,float arho){
 int st=vp[2]-vp[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);
 scale_copy_psv_k<<<(n+255)/256,256>>>(vp[1]+1,vs[1]+1,rho[1]+1,vpc[1]+1,vsc[1]+1,
  rhoc[1]+1,st,nx,ny,idx,idy,avp,avs,arho);cudaError_t e=cudaDeviceSynchronize();
 if(e!=cudaSuccess){fprintf(stderr,"FWI gradient scale/copy CUDA: %s\n",cudaGetErrorString(e));return -1;}return 0;}
__global__ static void pack_pcg_psv_k(float *v,float *vp,float *vs,float *rho,
 int st,int nx,int ny,int idx,int idy,int unpack){int q=blockIdx.x*blockDim.x+threadIdx.x,
 nxq=(nx+idx-1)/idx,nyq=(ny+idy-1)/idy,n=nxq*nyq;if(q>=n)return;
 int x=(q/nyq)*idx,y=(q%nyq)*idy,k=y*st+x;if(unpack){vp[k]=v[q];vs[k]=v[n+q];rho[k]=v[2*n+q];}
 else{v[q]=vp[k];v[n+q]=vs[k];v[2*n+q]=rho[k];}}
extern "C" int denise_gpu_pack_pcg_psv(float *v,float **vp,float **vs,float **rho,
 int nx,int ny,int idx,int idy,int unpack){int st=vp[2]-vp[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);
 pack_pcg_psv_k<<<(n+255)/256,256>>>(v+1,vp[1]+1,vs[1]+1,rho[1]+1,st,nx,ny,idx,idy,unpack);
 cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess){fprintf(stderr,"FWI PCG pack CUDA: %s\n",cudaGetErrorString(e));return -1;}return 0;}
__global__ static void pcg_sums_k(const float *cur,const float *old,const float *dir,
 int n,int method,float *num,float *den){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=n)return;
 int c=blockIdx.y,k=c*n+q;float a=cur[k],b=old[k],d=dir[k],delta=a-b,x=0,y=0;
 if(method==1){x=a*a;y=b*b;}else if(method==2){x=a*delta;y=b*b;}
 else if(method==3){x=-a*delta;y=d*delta;}else{x=-a*a;y=d*delta;}
 atomicAdd(num+c,x);atomicAdd(den+c,y);}
__global__ static void pcg_direction_k(const float *cur,float *old,float *dir,int n,
 const float *beta){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=n)return;int c=blockIdx.y,k=c*n+q;
 float a=cur[k];dir[k]=a+beta[c]*dir[k];old[k]=a;}
extern "C" int denise_gpu_pcg_update(float *cur,float *old,float *dir,int classes,
 int n,int method,float *beta){float *s=0;cudaError_t e=cudaMalloc(&s,2*classes*sizeof(float));if(e!=cudaSuccess)return -1;
 cudaMemsetAsync(s,0,2*classes*sizeof(float));dim3 grid((n+255)/256,classes);
 pcg_sums_k<<<grid,256>>>(cur+1,old+1,dir+1,n,method,s,s+classes);e=cudaDeviceSynchronize();
 float host[16];if(classes>8){cudaFree(s);return -1;}if(e==cudaSuccess)e=cudaMemcpy(host,s,2*classes*sizeof(float),cudaMemcpyDeviceToHost);
 if(e==cudaSuccess)for(int c=0;c<classes;c++){beta[c]=host[classes+c]!=0?host[c]/host[classes+c]:0;if(method==2&&beta[c]<0)beta[c]=0;}
 if(e==cudaSuccess)e=cudaMemcpy(s,beta,classes*sizeof(float),cudaMemcpyHostToDevice);
 if(e==cudaSuccess){pcg_direction_k<<<grid,256>>>(cur+1,old+1,dir+1,n,s);e=cudaDeviceSynchronize();}cudaFree(s);
 if(e!=cudaSuccess){fprintf(stderr,"FWI PCG CUDA: %s\n",cudaGetErrorString(e));return -1;}return 0;}
__global__ static void vec_dot_k(const float *a,const float *b,int n,float *sum){
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<n)atomicAdd(sum,a[q]*b[q]);}
__global__ static void vec_axpy_k(float *out,const float *x,int n,float a){
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<n)out[q]+=a*x[q];}
__global__ static void vec_scale_k(float *out,const float *x,int n,float a){
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<n)out[q]=a*x[q];}
static int gpu_dot(const float *a,const float *b,int n,float *dev,float *host){
 cudaMemsetAsync(dev,0,sizeof(float));vec_dot_k<<<(n+255)/256,256>>>(a,b,n,dev);
 cudaError_t e=cudaDeviceSynchronize();if(e==cudaSuccess)e=cudaMemcpy(host,dev,sizeof(float),cudaMemcpyDeviceToHost);return e==cudaSuccess?0:-1;}
extern "C" int denise_gpu_lbfgs_update(float *y,float *s,float *rho,float *alpha,
 float *q,float *r,int pointer,int hist,int n){float *d=0,nom,den;
 cudaError_t e=cudaMalloc(&d,sizeof(float));if(e!=cudaSuccess)return -1;
 int off=(pointer-1)*n;if(gpu_dot(y+1+off,s+1+off,n,d,&nom)||gpu_dot(y+1+off,y+1+off,n,d,&den)){cudaFree(d);return -1;}
 float gamma=nom/den;for(int k=0;k<hist;k++){off=k*n;if(gpu_dot(y+1+off,s+1+off,n,d,&nom)){cudaFree(d);return -1;}
  rho[k+1]=fabsf(nom)>0?1.0f/nom:0.0f;}
 for(int k=hist-1;k>=0;k--){off=k*n;if(gpu_dot(s+1+off,q+1,n,d,&nom)){cudaFree(d);return -1;}
  alpha[k+1]=rho[k+1]*nom;vec_axpy_k<<<(n+255)/256,256>>>(q+1,y+1+off,n,-alpha[k+1]);}
 vec_scale_k<<<(n+255)/256,256>>>(r+1,q+1,n,gamma);
 for(int k=0;k<hist;k++){off=k*n;if(gpu_dot(y+1+off,r+1,n,d,&nom)){cudaFree(d);return -1;}
  float beta=rho[k+1]*nom;vec_axpy_k<<<(n+255)/256,256>>>(r+1,s+1+off,n,alpha[k+1]-beta);}
 e=cudaDeviceSynchronize();cudaFree(d);if(e!=cudaSuccess){fprintf(stderr,"FWI L-BFGS CUDA: %s\n",cudaGetErrorString(e));return -1;}return 0;}
extern "C" int denise_gpu_fwi_store_ac(float **vx,float **vy,float **p,float **ux,float *frx,float *fry,float *fp,int nx,int ny,int idx,int idy,int off,int grad){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);store_ac_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,p[1]+1,ux[1]+1,frx,fry,fp,st,nx,ny,idx,idy,off,grad);return done("FWI AC store CUDA");}
extern "C" int denise_gpu_fwi_corr_ac(float **vx,float **vy,float **p,float **gl,float **gr,const float *frx,const float *fry,const float *fp,int nx,int ny,int idx,int idy,int off){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);corr_ac_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,p[1]+1,gl[1]+1,gr[1]+1,frx,fry,fp,st,nx,ny,idx,idy,off);return done("FWI AC correlate CUDA");}
extern "C" int denise_gpu_fwi_store_sh(float **vz,float **sxz,float **syz,float **uzx,float **uz,float *fr,float *fx,float *fy,int nx,int ny,int idx,int idy,int off,int grad){int st=vz[2]-vz[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);store_sh_k<<<(n+255)/256,256>>>(vz[1]+1,sxz[1]+1,syz[1]+1,uzx[1]+1,uz[1]+1,fr,fx,fy,st,nx,ny,idx,idy,off,grad);return done("FWI SH store CUDA");}
extern "C" int denise_gpu_fwi_corr_sh(float **vz,float **sxz,float **syz,float **gr,float **gu,const float *fr,const float *fx,const float *fy,float **rho,float **mu,int nx,int ny,int idx,int idy,int off,int invmat){int st=vz[2]-vz[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);corr_sh_k<<<(n+255)/256,256>>>(vz[1]+1,sxz[1]+1,syz[1]+1,gr[1]+1,gu[1]+1,fr,fx,fy,rho[1]+1,mu[1]+1,st,nx,ny,idx,idy,off,invmat);return done("FWI SH correlate CUDA");}
extern "C" int denise_gpu_fwi_store_psv(float **vx,float **vy,float **sxx,float **syy,float **sxy,float **ux,float **uy,float **uxy,float *frx,float *fry,float *fx,float *fy,float *fu,int nx,int ny,int idx,int idy,int off,int grad){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);store_psv_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,sxx[1]+1,syy[1]+1,sxy[1]+1,ux[1]+1,uy[1]+1,uxy[1]+1,frx,fry,fx,fy,fu,st,nx,ny,idx,idy,off,grad);return done("FWI P-SV store CUDA");}
extern "C" int denise_gpu_fwi_corr_psv(float **vx,float **vy,float **sxx,float **syy,float **sxy,float **gl,float **gm,float **gr,const float *frx,const float *fry,const float *fx,const float *fy,const float *fu,float **rho,float **vp,float **vs,int nx,int ny,int idx,int idy,int off,int grad,int invmat){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);corr_psv_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,sxx[1]+1,syy[1]+1,sxy[1]+1,gl[1]+1,gm[1]+1,gr[1]+1,frx,fry,fx,fy,fu,rho[1]+1,vp[1]+1,vs[1]+1,st,nx,ny,idx,idy,off,grad,invmat);return done("FWI P-SV correlate CUDA");}
extern "C" int denise_gpu_fwi_corr_aniso(float **vx,float **vy,float **sxx,float **syy,float **gl,float **gr,const float *frx,const float *fry,const float *fx,const float *fy,int nx,int ny,int idx,int idy,int off){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);corr_aniso_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,sxx[1]+1,syy[1]+1,gl[1]+1,gr[1]+1,frx,fry,fx,fy,st,nx,ny,idx,idy,off);return done("FWI anisotropic correlate CUDA");}
