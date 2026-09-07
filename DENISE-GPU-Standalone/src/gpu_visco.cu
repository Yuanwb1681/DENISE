#include <cuda_runtime.h>
#include <stdio.h>
#include "denise_gpu.h"

__global__ static void visc_psv_kernel(int nx,int ny,int st,int spx,int spy,
 float *vx,float *vy,float *ux,float *uy,float *uxy,float *sxx,float *syy,float *sxy,
 float *r,float *p,float *q,const float *fipjp,const float *fm,const float *gm,
 const float *bip,const float *bjm,const float *cip,const float *cjm,const float *dm,
 const float *em,const float *dip,const float *hc,const float *Kx,const float *ax,
 const float *bx,const float *Kxh,const float *axh,const float *bxh,const float *Ky,
 const float *ay,const float *by,const float *Kyh,const float *ayh,const float *byh,
 float *pvxx,float *pvyy,float *pvxy,float *pvyx,int fo,int fw,int fs,int bd,
 int npx,int npy,int posx,int posy,int mode,int grad,float dt,float dh,int L)
{
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=nx||y>=ny)return;
 float xx=0.0f,yy=0.0f,xy=0.0f,yx=0.0f;
 for(int m=1;m<=fo;m++){
  xx+=hc[m]*(vx[y*st+x+m-1]-vx[y*st+x-m]);
  yy+=hc[m]*(vy[(y+m-1)*st+x]-vy[(y-m)*st+x]);
  yx+=hc[m]*(vy[y*st+x+m]-vy[y*st+x-m+1]);
  xy+=hc[m]*(vx[(y+m)*st+x]-vx[(y-m+1)*st+x]);
 }
 xx/=dh;yy/=dh;xy/=dh;yx/=dh;int i=x+1,j=y+1;
 if(!bd&&posx==0&&i<=fw){float z=bx[i]*pvxx[y*spx+x]+ax[i]*xx;pvxx[y*spx+x]=z;xx=xx/Kx[i]+z;z=bxh[i]*pvyx[y*spx+x]+axh[i]*yx;pvyx[y*spx+x]=z;yx=yx/Kxh[i]+z;}
 if(!bd&&posx==npx-1&&i>=nx-fw+1){int h=i-nx+2*fw,z=h-1;float w=bx[h]*pvxx[y*spx+z]+ax[h]*xx;pvxx[y*spx+z]=w;xx=xx/Kx[h]+w;w=bxh[h]*pvyx[y*spx+z]+axh[h]*yx;pvyx[y*spx+z]=w;yx=yx/Kxh[h]+w;}
 if(posy==0&&!fs&&j<=fw){float z=by[j]*pvyy[y*spy+x]+ay[j]*yy;pvyy[y*spy+x]=z;yy=yy/Ky[j]+z;z=byh[j]*pvxy[y*spy+x]+ayh[j]*xy;pvxy[y*spy+x]=z;xy=xy/Kyh[j]+z;}
 if(posy==npy-1&&j>=ny-fw+1){int h=j-ny+2*fw,z=h-1;float w=by[h]*pvyy[z*spy+x]+ay[h]*yy;pvyy[z*spy+x]=w;yy=yy/Ky[h]+w;w=byh[h]*pvxy[z*spy+x]+ayh[h]*xy;pvxy[z*spy+x]=w;xy=xy/Kyh[h]+w;}
 int k=y*st+x,tk=k*L;float sr=0.0f,sp=0.0f,sq=0.0f;
 for(int l=0;l<L;l++){sr+=r[tk+l];sp+=p[tk+l];sq+=q[tk+l];}
 float sh=xy+yx,vol=xx+yy;
 sxy[k]+=fipjp[k]*sh+0.5f*dt*sr;
 sxx[k]+=gm[k]*vol-2.0f*fm[k]*yy+0.5f*dt*sp;
 syy[k]+=gm[k]*vol-2.0f*fm[k]*xx+0.5f*dt*sq;
 sr=sp=sq=0.0f;
 for(int l=0;l<L;l++){
  float nr=bip[l+1]*(r[tk+l]*cip[l+1]-dip[tk+l]*sh);
  float np=bjm[l+1]*(p[tk+l]*cjm[l+1]-em[tk+l]*vol+2.0f*dm[tk+l]*yy);
  float nq=bjm[l+1]*(q[tk+l]*cjm[l+1]-em[tk+l]*vol+2.0f*dm[tk+l]*xx);
  r[tk+l]=nr;p[tk+l]=np;q[tk+l]=nq;sr+=nr;sp+=np;sq+=nq;
 }
 sxy[k]+=0.5f*dt*sr;sxx[k]+=0.5f*dt*sp;syy[k]+=0.5f*dt*sq;
 if(mode==0&&grad==2){ux[k]=sp;uy[k]=sq;uxy[k]=sr;}
}

extern "C" int denise_gpu_update_s_visc_psv(int x1,int x2,int y1,int y2,
 float **vx,float **vy,float **ux,float **uy,float **uxy,float **sxx,float **syy,
 float **sxy,float ***r,float ***p,float ***q,float **fipjp,float **fm,float **gm,
 float *bip,float *bjm,float *cip,float *cjm,float ***dm,float ***em,float ***dip,
 const float *hc,const float *Kx,const float *ax,const float *bx,const float *Kxh,
 const float *axh,const float *bxh,const float *Ky,const float *ay,const float *by,
 const float *Kyh,const float *ayh,const float *byh,float **pvxx,float **pvyy,
 float **pvxy,float **pvyx,int mode,int fd,int fw,int fs,int bd,int npx,int npy,
 int posx,int posy,int grad,float dt,float dh,int L)
{
 int nx=x2-x1+1,ny=y2-y1+1,st=vx[y1+1]-vx[y1];
 int spx=pvxx[y1+1]-pvxx[y1],spy=pvyy[2]-pvyy[1];dim3 b(32,8),g((nx+31)/32,(ny+7)/8);
 visc_psv_kernel<<<g,b>>>(nx,ny,st,spx,spy,vx[y1]+x1,vy[y1]+x1,ux[y1]+x1,
  uy[y1]+x1,uxy[y1]+x1,sxx[y1]+x1,syy[y1]+x1,sxy[y1]+x1,r[y1][x1]+1,
  p[y1][x1]+1,q[y1][x1]+1,fipjp[y1]+x1,fm[y1]+x1,gm[y1]+x1,bip,bjm,cip,cjm,
  dm[y1][x1]+1,em[y1][x1]+1,dip[y1][x1]+1,hc,Kx,ax,bx,Kxh,axh,bxh,Ky,ay,by,
  Kyh,ayh,byh,pvxx[y1]+1,pvyy[1]+x1,pvxy[1]+x1,pvyx[y1]+1,fd/2,fw,fs,bd,
  npx,npy,posx,posy,mode,grad,dt,dh,L);
 cudaError_t e=cudaDeviceSynchronize();
 if(e!=cudaSuccess){fprintf(stderr,"visco P-SV stress CUDA: %s\n",cudaGetErrorString(e));return -1;}
 return 0;
}

__global__ static void visc_sh_kernel(int nx,int ny,int st,int spx,int spy,
 float *vz,float *uz,float *uzx,float *syz,float *sxz,float *r,float *q,
 const float *fipjp,const float *fm,const float *bip,const float *bjm,
 const float *cip,const float *cjm,const float *dm,const float *dip,
 const float *hc,const float *Kxh,const float *axh,const float *bxh,
 const float *Ky,const float *ay,const float *by,const float *Kyh,
 const float *ayh,const float *byh,float *pvzx,float *pvzy,int fo,int fw,
 int fs,int bd,int npx,int npy,int posx,int posy,int grad,float dt,float dh,int L)
{
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=nx||y>=ny)return;
 float zx=0.0f,zy=0.0f;
 for(int m=1;m<=fo;m++){
  zx+=hc[m]*(vz[y*st+x+m]-vz[y*st+x-m+1]);
  zy+=hc[m]*(vz[(y+m)*st+x]-vz[(y-m+1)*st+x]);
 }
 zx/=dh;zy/=dh;int i=x+1,j=y+1;
 if(fw>0){
  if(!bd&&posx==0&&i<=fw){float z=bxh[i]*pvzx[y*spx+x]+axh[i]*zx;pvzx[y*spx+x]=z;zx=zx/Kxh[i]+z;}
  if(!bd&&posx==npx-1&&i>=nx-fw+1){int h=i-nx+2*fw,z=h-1;float w=bxh[h]*pvzx[y*spx+z]+axh[h]*zx;pvzx[y*spx+z]=w;zx=zx/Kxh[h]+w;}
  if(posy==0&&!fs&&j<=fw){float z=by[j]*pvzy[y*spy+x]+ay[j]*zy;pvzy[y*spy+x]=z;zy=zy/Ky[j]+z;}
  if(posy==npy-1&&j>=ny-fw+1){int h=j-ny+2*fw,z=h-1;float w=byh[h]*pvzy[z*spy+x]+ayh[h]*zy;pvzy[z*spy+x]=w;zy=zy/Kyh[h]+w;}
 }
 int k=y*st+x,tk=k*L;float sr=0.0f,sq=0.0f;
 for(int l=0;l<L;l++){sr+=r[tk+l];sq+=q[tk+l];}
 sxz[k]+=fipjp[k]*zx+0.5f*dt*sr;syz[k]+=fm[k]*zy+0.5f*dt*sq;
 sr=sq=0.0f;
 for(int l=0;l<L;l++){
  float nr=bip[l+1]*(r[tk+l]*cip[l+1]-dip[tk+l]*zx);
  float nq=bjm[l+1]*(q[tk+l]*cjm[l+1]-dm[tk+l]*zy);
  r[tk+l]=nr;q[tk+l]=nq;sr+=nr;sq+=nq;
 }
 if(grad==2){uz[k]=zy;uzx[k]=zx;}
 sxz[k]+=0.5f*dt*sr;syz[k]+=0.5f*dt*sq;
}

extern "C" int denise_gpu_update_s_visc_sh(int x1,int x2,int y1,int y2,
 float **vz,float **uz,float **uzx,float **syz,float **sxz,float ***r,float ***q,
 float **fipjp,float **fm,float *bip,float *bjm,float *cip,float *cjm,float ***dm,
 float ***dip,const float *hc,const float *Kxh,const float *axh,const float *bxh,
 const float *Ky,const float *ay,const float *by,const float *Kyh,const float *ayh,
 const float *byh,float **pvzx,float **pvzy,int mode,int fd,int fw,int fs,int bd,
 int npx,int npy,int posx,int posy,int grad,float dt,float dh,int L)
{
 int nx=x2-x1+1,ny=y2-y1+1,st=vz[y1+1]-vz[y1];
 int spx=pvzx[y1+1]-pvzx[y1],spy=pvzy[2]-pvzy[1];dim3 b(32,8),g((nx+31)/32,(ny+7)/8);
 visc_sh_kernel<<<g,b>>>(nx,ny,st,spx,spy,vz[y1]+x1,uz[y1]+x1,uzx[y1]+x1,
  syz[y1]+x1,sxz[y1]+x1,r[y1][x1]+1,q[y1][x1]+1,fipjp[y1]+x1,
  fm[y1]+x1,bip,bjm,cip,cjm,dm[y1][x1]+1,dip[y1][x1]+1,hc,Kxh,axh,bxh,
  Ky,ay,by,Kyh,ayh,byh,pvzx[y1]+1,pvzy[1]+x1,fd/2,fw,fs,bd,npx,npy,
  posx,posy,grad,dt,dh,L);
 cudaError_t e=cudaDeviceSynchronize();
 if(e!=cudaSuccess){fprintf(stderr,"visco SH stress CUDA: %s\n",cudaGetErrorString(e));return -1;}
 return 0;
}
