#include <cuda_runtime.h>
#include <stdio.h>
#include "denise_gpu.h"

__global__ static void aniso(
    int nx, int ny, int st, int spx, int spy, float *vx, float *vy,
    float *ux, float *uy, float *uxy, float *sxx, float *syy, float *sxy,
    const float *a11, const float *a13, const float *a15, const float *a15h,
    const float *a33, const float *a35, const float *a35h, const float *a55,
    const float *hc, const float *Kx, const float *ax, const float *bx,
    const float *Kxh, const float *axh, const float *bxh, const float *Ky,
    const float *ay, const float *by, const float *Kyh, const float *ayh,
    const float *byh, float *pxx, float *pyy, float *pxy, float *pyx,
    int mode, int fo, int fw, int fs, int bd, int npx, int npy,
    int posx, int posy, int grad, float dt, float dh, int tilted)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=nx || y>=ny) return;
    float xx=0.0f, yy=0.0f, xy=0.0f, yx=0.0f;
    for(int m=1; m<=fo; ++m){
        xx += hc[m]*(vx[y*st+x+m-1]-vx[y*st+x-m]);
        yy += hc[m]*(vy[(y+m-1)*st+x]-vy[(y-m)*st+x]);
        yx += hc[m]*(vy[y*st+x+m]-vy[y*st+x-m+1]);
        xy += hc[m]*(vx[(y+m)*st+x]-vx[(y-m+1)*st+x]);
    }
    xx*=dt/dh; yy*=dt/dh; xy*=dt/dh; yx*=dt/dh;
    int i=x+1, j=y+1;
    if(!bd && posx==0 && i<=fw){
        float q=bx[i]*pxx[y*spx+x]+ax[i]*xx; pxx[y*spx+x]=q; xx=xx/Kx[i]+q;
        q=bxh[i]*pyx[y*spx+x]+axh[i]*yx; pyx[y*spx+x]=q; yx=yx/Kxh[i]+q;
    }
    if(!bd && posx==npx-1 && i>=nx-fw+1){
        int h=i-nx+2*fw, z=h-1;
        float q=bx[h]*pxx[y*spx+z]+ax[h]*xx; pxx[y*spx+z]=q; xx=xx/Kx[h]+q;
        q=bxh[h]*pyx[y*spx+z]+axh[h]*yx; pyx[y*spx+z]=q; yx=yx/Kxh[h]+q;
    }
    if(posy==0 && !fs && j<=fw){
        float q=by[j]*pyy[y*spy+x]+ay[j]*yy; pyy[y*spy+x]=q; yy=yy/Ky[j]+q;
        q=byh[j]*pxy[y*spy+x]+ayh[j]*xy; pxy[y*spy+x]=q; xy=xy/Kyh[j]+q;
    }
    if(posy==npy-1 && j>=ny-fw+1){
        int h=j-ny+2*fw, z=h-1;
        float q=by[h]*pyy[z*spy+x]+ay[h]*yy; pyy[z*spy+x]=q; yy=yy/Ky[h]+q;
        q=byh[h]*pxy[z*spy+x]+ayh[h]*xy; pxy[z*spy+x]=q; xy=xy/Kyh[h]+q;
    }
    int k=y*st+x;
    float sh=xy+yx, ox, oy, oz;
    if(tilted){
        ox=a11[k]*xx+a13[k]*yy+a15h[k]*sh;
        oy=a13[k]*xx+a33[k]*yy+a35h[k]*sh;
        oz=a15[k]*xx+a35[k]*yy+a55[k]*sh;
    }else{
        ox=a11[k]*xx+a13[k]*yy;
        oy=a13[k]*xx+a33[k]*yy;
        oz=a55[k]*sh;
    }
    if(mode==0 && grad==2 && !tilted){ ux[k]=ox/dt; uy[k]=oy/dt; uxy[k]=oz/dt; }
    sxx[k]+=ox; syy[k]+=oy; sxy[k]+=oz;
}

extern "C" int denise_gpu_update_s_aniso(
    int x1,int x2,int y1,int y2,float **vx,float **vy,float **ux,float **uy,
    float **uxy,float **sxx,float **syy,float **sxy,float **a11,float **a13,
    float **a15,float **a15h,float **a33,float **a35,float **a35h,float **a55,
    const float *hc,const float *Kx,const float *ax,const float *bx,
    const float *Kxh,const float *axh,const float *bxh,const float *Ky,
    const float *ay,const float *by,const float *Kyh,const float *ayh,
    const float *byh,float **pxx,float **pyy,float **pxy,float **pyx,int mode,
    int fd,int fw,int fs,int bd,int npx,int npy,int posx,int posy,int grad,
    float dt,float dh,int tilted)
{
    int nx=x2-x1+1, ny=y2-y1+1;
    int st=vx[y1+1]-vx[y1], spx=pxx[y1+1]-pxx[y1], spy=pyy[2]-pyy[1];
    dim3 b(32,8), g((nx+31)/32,(ny+7)/8);
    aniso<<<g,b>>>(nx,ny,st,spx,spy,vx[y1]+x1,vy[y1]+x1,ux[y1]+x1,
        uy[y1]+x1,uxy[y1]+x1,sxx[y1]+x1,syy[y1]+x1,sxy[y1]+x1,
        a11[y1]+x1,a13[y1]+x1,a15?a15[y1]+x1:0,a15h?a15h[y1]+x1:0,
        a33[y1]+x1,a35?a35[y1]+x1:0,a35h?a35h[y1]+x1:0,a55[y1]+x1,
        hc,Kx,ax,bx,Kxh,axh,bxh,Ky,ay,by,Kyh,ayh,byh,pxx[y1]+1,
        pyy[1]+x1,pxy[1]+x1,pyx[y1]+1,mode,fd/2,fw,fs,bd,npx,npy,
        posx,posy,grad,dt,dh,tilted);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ fprintf(stderr,"anisotropic stress CUDA: %s\n",cudaGetErrorString(e)); return -1; }
    return 0;
}
