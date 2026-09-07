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
extern "C" int denise_gpu_fwi_store_ac(float **vx,float **vy,float **p,float **ux,float *frx,float *fry,float *fp,int nx,int ny,int idx,int idy,int off,int grad){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);store_ac_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,p[1]+1,ux[1]+1,frx,fry,fp,st,nx,ny,idx,idy,off,grad);return done("FWI AC store CUDA");}
extern "C" int denise_gpu_fwi_corr_ac(float **vx,float **vy,float **p,float **gl,float **gr,const float *frx,const float *fry,const float *fp,int nx,int ny,int idx,int idy,int off){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);corr_ac_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,p[1]+1,gl[1]+1,gr[1]+1,frx,fry,fp,st,nx,ny,idx,idy,off);return done("FWI AC correlate CUDA");}
extern "C" int denise_gpu_fwi_store_sh(float **vz,float **sxz,float **syz,float **uzx,float **uz,float *fr,float *fx,float *fy,int nx,int ny,int idx,int idy,int off,int grad){int st=vz[2]-vz[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);store_sh_k<<<(n+255)/256,256>>>(vz[1]+1,sxz[1]+1,syz[1]+1,uzx[1]+1,uz[1]+1,fr,fx,fy,st,nx,ny,idx,idy,off,grad);return done("FWI SH store CUDA");}
extern "C" int denise_gpu_fwi_corr_sh(float **vz,float **sxz,float **syz,float **gr,float **gu,const float *fr,const float *fx,const float *fy,float **rho,float **mu,int nx,int ny,int idx,int idy,int off,int invmat){int st=vz[2]-vz[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);corr_sh_k<<<(n+255)/256,256>>>(vz[1]+1,sxz[1]+1,syz[1]+1,gr[1]+1,gu[1]+1,fr,fx,fy,rho[1]+1,mu[1]+1,st,nx,ny,idx,idy,off,invmat);return done("FWI SH correlate CUDA");}
extern "C" int denise_gpu_fwi_store_psv(float **vx,float **vy,float **sxx,float **syy,float **sxy,float **ux,float **uy,float **uxy,float *frx,float *fry,float *fx,float *fy,float *fu,int nx,int ny,int idx,int idy,int off,int grad){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);store_psv_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,sxx[1]+1,syy[1]+1,sxy[1]+1,ux[1]+1,uy[1]+1,uxy[1]+1,frx,fry,fx,fy,fu,st,nx,ny,idx,idy,off,grad);return done("FWI P-SV store CUDA");}
extern "C" int denise_gpu_fwi_corr_psv(float **vx,float **vy,float **sxx,float **syy,float **sxy,float **gl,float **gm,float **gr,const float *frx,const float *fry,const float *fx,const float *fy,const float *fu,float **rho,float **vp,float **vs,int nx,int ny,int idx,int idy,int off,int grad,int invmat){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);corr_psv_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,sxx[1]+1,syy[1]+1,sxy[1]+1,gl[1]+1,gm[1]+1,gr[1]+1,frx,fry,fx,fy,fu,rho[1]+1,vp[1]+1,vs[1]+1,st,nx,ny,idx,idy,off,grad,invmat);return done("FWI P-SV correlate CUDA");}
extern "C" int denise_gpu_fwi_corr_aniso(float **vx,float **vy,float **sxx,float **syy,float **gl,float **gr,const float *frx,const float *fry,const float *fx,const float *fy,int nx,int ny,int idx,int idy,int off){int st=vx[2]-vx[1],n=((nx+idx-1)/idx)*((ny+idy-1)/idy);corr_aniso_k<<<(n+255)/256,256>>>(vx[1]+1,vy[1]+1,sxx[1]+1,syy[1]+1,gl[1]+1,gr[1]+1,frx,fry,fx,fy,st,nx,ny,idx,idy,off);return done("FWI anisotropic correlate CUDA");}
