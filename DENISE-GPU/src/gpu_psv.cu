#include <cuda_runtime.h>
#include <stdio.h>
#include "denise_gpu.h"

__global__ static void stress_psv(int nx,int ny,int svx,int svy,int so,
 int spx,int spy,float *vx,float *vy,float *ux,float *uy,float *uxy,
 float *sxx,float *syy,float *sxy,const float *pi,const float *mu,
 const float *muh,const float *rho,const float *hc,const float *Kx,
 const float *ax,const float *bx,const float *Kxh,const float *axh,
 const float *bxh,const float *Ky,const float *ay,const float *by,
 const float *Kyh,const float *ayh,const float *byh,float *pvxx,
 float *pvyy,float *pvxy,float *pvyx,int mode,int fdoh,int invmat,int fw,
 int free_surface,int boundary,int npx,int npy,int posx,int posy,int grad,
 float dt,float dh)
{
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=nx||y>=ny)return; float vxx=0,vyy=0,vxy=0,vyx=0;
 for(int m=1;m<=fdoh;m++){
  vxx+=hc[m]*(vx[y*svx+x+m-1]-vx[y*svx+x-m]);
  vyy+=hc[m]*(vy[(y+m-1)*svy+x]-vy[(y-m)*svy+x]);
  vyx+=hc[m]*(vy[y*svy+x+m]-vy[y*svy+x-m+1]);
  vxy+=hc[m]*(vx[(y+m)*svx+x]-vx[(y-m+1)*svx+x]);
 }
 vxx*=dt/dh;vyy*=dt/dh;vxy*=dt/dh;vyx*=dt/dh;
 int i=x+1,j=y+1;
 if(!boundary&&posx==0&&i<=fw){float q=bx[i]*pvxx[y*spx+x]+ax[i]*vxx;pvxx[y*spx+x]=q;vxx=vxx/Kx[i]+q;q=bxh[i]*pvyx[y*spx+x]+axh[i]*vyx;pvyx[y*spx+x]=q;vyx=vyx/Kxh[i]+q;}
 if(!boundary&&posx==npx-1&&i>=nx-fw+1){int h=i-nx+2*fw,z=h-1;float q=bx[h]*pvxx[y*spx+z]+ax[h]*vxx;pvxx[y*spx+z]=q;vxx=vxx/Kx[h]+q;q=bxh[h]*pvyx[y*spx+z]+axh[h]*vyx;pvyx[y*spx+z]=q;vyx=vyx/Kxh[h]+q;}
 if(posy==0&&!free_surface&&j<=fw){float q=by[j]*pvyy[y*spy+x]+ay[j]*vyy;pvyy[y*spy+x]=q;vyy=vyy/Ky[j]+q;q=byh[j]*pvxy[y*spy+x]+ayh[j]*vxy;pvxy[y*spy+x]=q;vxy=vxy/Kyh[j]+q;}
 if(posy==npy-1&&j>=ny-fw+1){int h=j-ny+2*fw,z=h-1;float q=by[h]*pvyy[z*spy+x]+ay[h]*vyy;pvyy[z*spy+x]=q;vyy=vyy/Ky[h]+q;q=byh[h]*pvxy[z*spy+x]+ayh[h]*vxy;pvxy[z*spy+x]=q;vxy=vxy/Kyh[h]+q;}
 int k=y*so+x;float f,g;if(invmat==3){f=mu[k];g=pi[k];}else{f=rho[k]*mu[k]*mu[k];g=rho[k]*(pi[k]*pi[k]-2.0f*mu[k]*mu[k]);}float sh=muh[k]*(vyx+vxy),xx=g*(vxx+vyy)+2.0f*f*vxx,yy=g*(vxx+vyy)+2.0f*f*vyy;
 if(mode==0&&grad==2){ux[k]=xx/dt;uy[k]=yy/dt;uxy[k]=sh/dt;}sxy[k]+=sh;sxx[k]+=xx;syy[k]+=yy;
}

extern "C" int denise_gpu_update_s_psv(int nx1,int nx2,int ny1,int ny2,
 float **vx,float **vy,float **ux,float **uy,float **uxy,float **sxx,float **syy,
 float **sxy,float **pi,float **u,float **uipjp,float **rho,const float *hc,
 const float *Kx,const float *ax,const float *bx,const float *Kxh,const float *axh,
 const float *bxh,const float *Ky,const float *ay,const float *by,const float *Kyh,
 const float *ayh,const float *byh,float **pvxx,float **pvyy,float **pvxy,
 float **pvyx,int mode,int fdorder,int invmat,int fw,int fs,int boundary,int npx,
 int npy,int posx,int posy,int grad,float dt,float dh)
{
 int nx=nx2-nx1+1,ny=ny2-ny1+1,svx=vx[ny1+1]-vx[ny1],svy=vy[ny1+1]-vy[ny1],so=sxx[ny1+1]-sxx[ny1],spx=pvxx[ny1+1]-pvxx[ny1],spy=pvyy[2]-pvyy[1];
 dim3 b(32,8),g((nx+31)/32,(ny+7)/8);stress_psv<<<g,b>>>(nx,ny,svx,svy,so,spx,spy,vx[ny1]+nx1,vy[ny1]+nx1,ux[ny1]+nx1,uy[ny1]+nx1,uxy[ny1]+nx1,sxx[ny1]+nx1,syy[ny1]+nx1,sxy[ny1]+nx1,pi[ny1]+nx1,u[ny1]+nx1,uipjp[ny1]+nx1,rho[ny1]+nx1,hc,Kx,ax,bx,Kxh,axh,bxh,Ky,ay,by,Kyh,ayh,byh,pvxx[ny1]+1,pvyy[1]+nx1,pvxy[1]+nx1,pvyx[ny1]+1,mode,fdorder/2,invmat,fw,fs,boundary,npx,npy,posx,posy,grad,dt,dh);
 cudaError_t e=cudaPeekAtLastError();if(e!=cudaSuccess){fprintf(stderr,"DENISE GPU PSV stress launch failed: %s\n",cudaGetErrorString(e));return -1;}return 0;
}

__global__ static void velocity_psv(int nx,int ny,int so,int sv,int spx,int spy,
 float *vx,float *vy,float *vxp,float *vyp,const float *sxx,const float *syy,
 const float *sxy,const float *rip,const float *rjp,const float *hc,
 const float *Kx,const float *ax,const float *bx,const float *Kxh,const float *axh,
 const float *bxh,const float *Ky,const float *ay,const float *by,const float *Kyh,
 const float *ayh,const float *byh,float *pxx,float *pyy,float *pxyy,float *pxyx,
 int sw,int fdoh,int fw,int fs,int boundary,int npx,int npy,int posx,int posy,
 int grad,float dt,float dh)
{
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=nx||y>=ny)return;
 float sxx_x=0,syy_y=0,sxy_y=0,sxy_x=0;for(int m=1;m<=fdoh;m++){sxx_x+=hc[m]*(sxx[y*so+x+m]-sxx[y*so+x-m+1]);sxy_x+=hc[m]*(sxy[y*so+x+m-1]-sxy[y*so+x-m]);sxy_y+=hc[m]*(sxy[(y+m-1)*so+x]-sxy[(y-m)*so+x]);syy_y+=hc[m]*(syy[(y+m)*so+x]-syy[(y-m+1)*so+x]);}
 int i=x+1,j=y+1;if(!boundary&&posx==0&&i<=fw){float z=bxh[i]*pxx[y*spx+x]+axh[i]*sxx_x;pxx[y*spx+x]=z;sxx_x=sxx_x/Kxh[i]+z;z=bx[i]*pxyx[y*spx+x]+ax[i]*sxy_x;pxyx[y*spx+x]=z;sxy_x=sxy_x/Kx[i]+z;}
 if(!boundary&&posx==npx-1&&i>=nx-fw+1){int h=i-nx+2*fw,k=h-1;float z=bxh[h]*pxx[y*spx+k]+axh[h]*sxx_x;pxx[y*spx+k]=z;sxx_x=sxx_x/Kxh[h]+z;z=bx[h]*pxyx[y*spx+k]+ax[h]*sxy_x;pxyx[y*spx+k]=z;sxy_x=sxy_x/Kx[h]+z;}
 if(posy==0&&!fs&&j<=fw){float z=byh[j]*pyy[y*spy+x]+ayh[j]*syy_y;pyy[y*spy+x]=z;syy_y=syy_y/Kyh[j]+z;z=by[j]*pxyy[y*spy+x]+ay[j]*sxy_y;pxyy[y*spy+x]=z;sxy_y=sxy_y/Ky[j]+z;}
 if(posy==npy-1&&j>=ny-fw+1){int h=j-ny+2*fw,k=h-1;float z=byh[h]*pyy[k*spy+x]+ayh[h]*syy_y;pyy[k*spy+x]=z;syy_y=syy_y/Kyh[h]+z;z=by[h]*pxyy[k*spy+x]+ay[h]*sxy_y;pxyy[k*spy+x]=z;sxy_y=sxy_y/Ky[h]+z;}
 int k=y*sv+x;float xx=rip[k]*(sxx_x+sxy_y)/dh,yy=rjp[k]*(sxy_x+syy_y)/dh;if(grad==1){if(sw==0){vxp[k]=xx;vyp[k]=yy;}else{vxp[k]+=dt*vx[k];vyp[k]+=dt*vy[k];}}else if(grad==2){if(sw==0){vxp[k]=xx;vyp[k]=yy;}else{vxp[k]=vx[k];vyp[k]=vy[k];}}vx[k]+=dt*xx;vy[k]+=dt*yy;
}

extern "C" int denise_gpu_update_v_psv(int nx1,int nx2,int ny1,int ny2,float **vx,float **vy,float **vxp,float **vyp,float **sxx,float **syy,float **sxy,float **rip,float **rjp,const float *hc,const float *Kx,const float *ax,const float *bx,const float *Kxh,const float *axh,const float *bxh,const float *Ky,const float *ay,const float *by,const float *Kyh,const float *ayh,const float *byh,float **pxx,float **pyy,float **pxyy,float **pxyx,int sw,int fdorder,int fw,int fs,int boundary,int npx,int npy,int posx,int posy,int grad,float dt,float dh)
{
 int nx=nx2-nx1+1,ny=ny2-ny1+1,sv=vx[ny1+1]-vx[ny1],so=sxx[ny1+1]-sxx[ny1],spx=pxx[ny1+1]-pxx[ny1],spy=pyy[2]-pyy[1];dim3 b(32,8),g((nx+31)/32,(ny+7)/8);velocity_psv<<<g,b>>>(nx,ny,so,sv,spx,spy,vx[ny1]+nx1,vy[ny1]+nx1,vxp[ny1]+nx1,vyp[ny1]+nx1,sxx[ny1]+nx1,syy[ny1]+nx1,sxy[ny1]+nx1,rip[ny1]+nx1,rjp[ny1]+nx1,hc,Kx,ax,bx,Kxh,axh,bxh,Ky,ay,by,Kyh,ayh,byh,pxx[ny1]+1,pyy[1]+nx1,pxyy[1]+nx1,pxyx[ny1]+1,sw,fdorder/2,fw,fs,boundary,npx,npy,posx,posy,grad,dt,dh);cudaError_t e=cudaPeekAtLastError();if(e!=cudaSuccess){fprintf(stderr,"DENISE GPU PSV velocity launch failed: %s\n",cudaGetErrorString(e));return -1;}return 0;
}

__global__ static void source_psv(int nt,int ntotal,float dt,float *sxx,float *syy,
 int ss,const float *sx,const float *sy,const float *sig,int si,int n,int sw,int qt)
{int l=blockIdx.x*blockDim.x+threadIdx.x;if(l>=n||(sw&&qt<4))return;int i=(int)sx[l],j=(int)sy[l];float a;if(sw)a=sig[l*si+nt];else if(nt==1)a=sig[l*si+nt+1]/dt;else if(nt<ntotal)a=(sig[l*si+nt+1]-sig[l*si+nt-1])/dt;else a=-sig[l*si+nt-1]/dt;atomicAdd(sxx+j*ss+i,a);atomicAdd(syy+j*ss+i,a);}
extern "C" int denise_gpu_psource_psv(int nt,int ntotal,float dt,float **sxx,float **syy,float **src,float **sig,int n,int sw,int qt)
{if(n<=0)return 0;int ss=sxx[1]-sxx[0],si=sig[2]-sig[1];source_psv<<<(n+127)/128,128>>>(nt,ntotal,dt,sxx[0],syy[0],ss,src[1]+1,src[2]+1,sig[1],si,n,sw,qt);cudaError_t e=cudaPeekAtLastError();if(e!=cudaSuccess){fprintf(stderr,"DENISE GPU PSV source launch failed: %s\n",cudaGetErrorString(e));return -1;}return 0;}

__global__ static void vsource_psv(int nt,float *vx,float *vy,int stride,
 const float *sx,const float *sy,const float *angle,const float *stype,
 const float *sigx,const float *sigy,int sigstride,int n,int sw,int qt)
{
 int l=blockIdx.x*blockDim.x+threadIdx.x;if(l>=n)return;
 int i=(int)sx[l],j=(int)sy[l],k=j*stride+i;
 if(sw){
  if(qt==1){atomicAdd(vx+k,sigx[l*sigstride+nt]);atomicAdd(vy+k,sigy[l*sigstride+nt]);}
  if(qt==2||qt==6||qt==7)atomicAdd(vy+k,sigy[l*sigstride+nt]);
  if(qt==3||qt==5||qt==7)atomicAdd(vx+k,sigx[l*sigstride+nt]);
 }else{
  int t=(int)stype[l];float a=sigy[l*sigstride+nt];
  if(t==2)atomicAdd(vx+k,a);else if(t==3)atomicAdd(vy+k,a);else if(t==4){float r=angle[l]*3.14159265358979323846f/180.0f;atomicAdd(vx+k,-sinf(r)*a);atomicAdd(vy+k,cosf(r)*a);}
 }
}
extern "C" int denise_gpu_vsource_psv(int nt,float **vx,float **vy,float **src,
 float **sigx,float **sigy,int n,int sw,int qt)
{
 if(n<=0)return 0;int st=vx[1]-vx[0],ss=sigx[2]-sigx[1];
 const float *angle=sw ? src[1]+1 : src[7]+1;
 const float *stype=sw ? src[1]+1 : src[8]+1;
 vsource_psv<<<(n+127)/128,128>>>(nt,vx[0],vy[0],st,src[1]+1,src[2]+1,
  angle,stype,sigx[1],sigy[1],ss,n,sw,qt);
 cudaError_t e=cudaPeekAtLastError();if(e!=cudaSuccess){fprintf(stderr,"DENISE GPU PSV force source launch failed: %s\n",cudaGetErrorString(e));return -1;}return 0;
}

__global__ static void surface_psv(int nx,int fdoh,int invmat,int fw,int boundary,
 int npx,int posx,float dt,float dh,float *vx,float *vy,float *sxx,float *syy,
 float *sxy,const float *pi,const float *mu,const float *rho,int stride,
 const float *hc,const float *Kx,const float *ax,const float *bx,float *psi)
{int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=nx)return;int i=x+1;syy[i]=0;float vxx=0,vyy=0;for(int m=1;m<=fdoh;m++){syy[-m*stride+i]=-syy[m*stride+i];sxy[-m*stride+i]=-sxy[(m-1)*stride+i];vxx+=hc[m]*(vx[i+m-1]-vx[i-m]);vyy+=hc[m]*(vy[(m-1)*stride+i]-vy[-m*stride+i]);}vxx/=dh;vyy/=dh;if(!boundary&&posx==0&&i<=fw){float q=bx[i]*psi[x]+ax[i]*vxx;psi[x]=q;vxx=vxx/Kx[i]+q;}if(!boundary&&posx==npx-1&&i>=nx-fw+1){int h=i-nx+2*fw,z=h-1;float q=bx[h]*psi[z]+ax[h]*vxx;psi[z]=q;vxx=vxx/Kx[h]+q;}float f,g;if(invmat==3){f=2*mu[i];g=pi[i];}else{f=2*rho[i]*mu[i]*mu[i];g=rho[i]*(pi[i]*pi[i]-2*mu[i]*mu[i]);}sxx[i]+=-dt*((g*g)/(g+f)*vxx+g*vyy);}
extern "C" int denise_gpu_surface_psv(int d,int nx,int fo,int im,int fw,int bd,int npx,int px,float dt,float dh,float **vx,float **vy,float **sxx,float **syy,float **sxy,float **pi,float **u,float **rho,const float *hc,const float *Kx,const float *ax,const float *bx,float **psi)
{int st=vx[d+1]-vx[d];surface_psv<<<(nx+127)/128,128>>>(nx,fo/2,im,fw,bd,npx,px,dt,dh,vx[d],vy[d],sxx[d],syy[d],sxy[d],pi[d],u[d],rho[d],st,hc,Kx,ax,bx,psi[d]+1);cudaError_t e=cudaPeekAtLastError();if(e!=cudaSuccess){fprintf(stderr,"DENISE GPU PSV surface launch failed: %s\n",cudaGetErrorString(e));return -1;}return 0;}

__global__ static void seismo_psv(int sample,int n,const int *rx,const int *ry,
 float *ovx,int xovx,float *ovy,int xovy,float *op,int xop,float *oc,int xoc,
 float *od,int xod,const float *vx,const float *vy,const float *sxx,
 const float *syy,const float *pi,const float *mu,const float *rho,int stride,
 const float *hc,int type,int fdoh,int invmat,float dt,float dh)
{int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n)return;int i=rx[t],j=ry[t],o=sample;
 if(type==1||type==4){ovx[t*xovx+o]=vx[j*stride+i];ovy[t*xovy+o]=vy[j*stride+i];}
 if(type==2||type==4)op[t*xop+o]=-sxx[j*stride+i]-syy[j*stride+i];
 if(type==3||type==4){float xx=0,yy=0,yx=0,xy=0;for(int m=1;m<=fdoh;m++){xx+=hc[m]*(vx[j*stride+i+m-1]-vx[j*stride+i-m]);yy+=hc[m]*(vy[(j+m-1)*stride+i]-vy[(j-m)*stride+i]);yx+=hc[m]*(vy[j*stride+i+m]-vy[j*stride+i-m+1]);xy+=hc[m]*(vx[(j+m)*stride+i]-vx[(j-m+1)*stride+i]);}xx*=dt/dh;yy*=dt/dh;yx*=dt/dh;xy*=dt/dh;float a,b;if(invmat==1){a=mu[j*stride+i]*mu[j*stride+i]*rho[j*stride+i];b=pi[j*stride+i]*pi[j*stride+i]*rho[j*stride+i];}else{a=mu[j*stride+i];b=pi[j*stride+i];}od[t*xod+o]=(xx+yy)*sqrtf(b);oc[t*xoc+o]=(xy-yx)*sqrtf(a);}}

extern "C" int denise_gpu_seismo_psv(int smp,int n,int **rec,float **ovx,
 float **ovy,float **op,float **oc,float **od,float **vx,float **vy,float **sxx,
 float **syy,float **pi,float **u,float **rho,const float *hc,int type,int fo,
 int im,float dt,float dh)
{if(n<=0)return 0;
#define B(a) ((a)?(a)[1]:NULL)
#define X(a) ((a)?(int)((a)[2]-(a)[1]):0)
 int st=vx[1]-vx[0];seismo_psv<<<(n+127)/128,128>>>(smp,n,rec[1]+1,rec[2]+1,
 B(ovx),X(ovx),B(ovy),X(ovy),B(op),X(op),B(oc),X(oc),B(od),X(od),vx[0],vy[0],
 sxx[0],syy[0],pi[0],u[0],rho[0],st,hc,type,fo/2,im,dt,dh);
#undef B
#undef X
 cudaError_t e=cudaPeekAtLastError();if(e!=cudaSuccess){fprintf(stderr,"DENISE GPU PSV receiver launch failed: %s\n",cudaGetErrorString(e));return -1;}return 0;}
