#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "denise_gpu.h"

static int selected_device = -1;
struct managed_block { void *ptr; size_t bytes; };
static managed_block *managed_blocks = NULL;
static size_t managed_count = 0, managed_capacity = 0;

static int local_rank_from_environment(int fallback)
{
    const char *names[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID"
    };
    for (unsigned int i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        const char *value = getenv(names[i]);
        if (value && *value) return atoi(value);
    }
    return fallback;
}

extern "C" int denise_gpu_init(int world_rank)
{
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        fprintf(stderr, "DENISE GPU rank %d: no CUDA device: %s\n",
                world_rank, cudaGetErrorString(status));
        return -1;
    }

    selected_device = local_rank_from_environment(world_rank) % count;
    status = cudaSetDevice(selected_device);
    if (status != cudaSuccess) {
        fprintf(stderr, "DENISE GPU rank %d: cudaSetDevice(%d) failed: %s\n",
                world_rank, selected_device, cudaGetErrorString(status));
        return -1;
    }

    cudaDeviceProp prop;
    status = cudaGetDeviceProperties(&prop, selected_device);
    if (status != cudaSuccess) return -1;
    fprintf(stdout,
            "DENISE GPU rank %d uses CUDA device %d: %s (CC %d.%d, %.0f MiB)\n",
            world_rank, selected_device, prop.name, prop.major, prop.minor,
            (double)prop.totalGlobalMem / (1024.0 * 1024.0));
    return 0;
}

extern "C" void denise_gpu_finalize(void)
{
    if (selected_device >= 0) cudaDeviceSynchronize();
}

extern "C" int denise_gpu_sync(const char *where)
{
    cudaError_t status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return 0;
    fprintf(stderr, "DENISE GPU%s%s: %s\n", where ? " " : "",
            where ? where : "synchronize failed", cudaGetErrorString(status));
    return -1;
}

extern "C" void *denise_gpu_managed_alloc(unsigned long bytes)
{
    void *ptr = NULL;
    cudaError_t status = cudaMallocManaged(&ptr, (size_t)bytes, cudaMemAttachGlobal);
    if (status != cudaSuccess) {
        fprintf(stderr, "DENISE GPU: cudaMallocManaged(%lu) failed: %s\n",
                bytes, cudaGetErrorString(status));
        return NULL;
    }
    cudaMemset(ptr, 0, (size_t)bytes);
    if (managed_count == managed_capacity) {
        size_t next = managed_capacity ? managed_capacity * 2 : 256;
        managed_block *blocks = (managed_block *)realloc(managed_blocks,
                                                         next * sizeof(*blocks));
        if (!blocks) { cudaFree(ptr); return NULL; }
        managed_blocks = blocks; managed_capacity = next;
    }
    managed_blocks[managed_count++] = {ptr, (size_t)bytes};
    return ptr;
}

extern "C" void denise_gpu_managed_free(void *ptr)
{
    if (!ptr) return;
    for (size_t i = 0; i < managed_count; ++i) if (managed_blocks[i].ptr == ptr) {
        managed_blocks[i] = managed_blocks[--managed_count]; break;
    }
    cudaFree(ptr);
}

extern "C" int denise_gpu_prefetch_all(void)
{
    if (selected_device < 0) return -1;
    for (size_t i = 0; i < managed_count; ++i) {
        cudaError_t e = cudaMemPrefetchAsync(managed_blocks[i].ptr,
                                             managed_blocks[i].bytes,
                                             selected_device, 0);
        if (e == cudaErrorInvalidDevice || e == cudaErrorNotSupported) {
            cudaGetLastError(); return 0;
        }
        if (e != cudaSuccess) {
            fprintf(stderr,"DENISE GPU prefetch failed: %s\n",cudaGetErrorString(e));
            return -1;
        }
    }
    return denise_gpu_sync("after managed-memory prefetch");
}

__global__ static void update_s_acoustic_kernel(
    int nx, int ny, int sx_vx, int sx_vy, int sx_ux, int sx_p, int sx_pi,
    int sx_rho, int sx_psix, int sx_psiy, float *vx, float *vy, float *ux,
    float *p, const float *pi, const float *rho, const float *hc,
    const float *K_x, const float *a_x, const float *b_x,
    const float *K_y, const float *a_y, const float *b_y,
    float *psi_vxx, float *psi_vyy, int mode, int fdoh, int invmat1, int fw,
    int free_surf, int boundary, int nprocx, int nprocy, int posx, int posy,
    int grad_form, float dt, float dh)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    float vxx = 0.0f, vyy = 0.0f;
    for (int m = 1; m <= fdoh; ++m) {
        vxx += hc[m] * (vx[iy * sx_vx + ix + m - 1] -
                        vx[iy * sx_vx + ix - m]);
        vyy += hc[m] * (vy[(iy + m - 1) * sx_vy + ix] -
                        vy[(iy - m) * sx_vy + ix]);
    }
    vxx *= dt / dh;
    vyy *= dt / dh;

    int i = ix + 1;
    int j = iy + 1;
    if (!boundary && posx == 0 && i <= fw) {
        float q = b_x[i] * psi_vxx[iy * sx_psix + ix] + a_x[i] * vxx;
        psi_vxx[iy * sx_psix + ix] = q;
        vxx = vxx / K_x[i] + q;
    }
    if (!boundary && posx == nprocx - 1 && i >= nx - fw + 1) {
        int h = i - nx + 2 * fw;
        int qx = h - 1;
        float q = b_x[h] * psi_vxx[iy * sx_psix + qx] + a_x[h] * vxx;
        psi_vxx[iy * sx_psix + qx] = q;
        vxx = vxx / K_x[h] + q;
    }
    if (posy == 0 && !free_surf && j <= fw) {
        float q = b_y[j] * psi_vyy[iy * sx_psiy + ix] + a_y[j] * vyy;
        psi_vyy[iy * sx_psiy + ix] = q;
        vyy = vyy / K_y[j] + q;
    }
    if (posy == nprocy - 1 && j >= ny - fw + 1) {
        int h = j - ny + 2 * fw;
        int qy = h - 1;
        float q = b_y[h] * psi_vyy[qy * sx_psiy + ix] + a_y[h] * vyy;
        psi_vyy[qy * sx_psiy + ix] = q;
        vyy = vyy / K_y[h] + q;
    }

    float g = invmat1 == 3 ? pi[iy * sx_pi + ix]
                           : rho[iy * sx_rho + ix] * pi[iy * sx_pi + ix] *
                             pi[iy * sx_pi + ix];
    if (mode == 0 && grad_form == 2)
        ux[iy * sx_ux + ix] = g * (vxx + vyy) / dt;
    p[iy * sx_p + ix] += g * (vxx + vyy);
}

extern "C" int denise_gpu_update_s_acoustic(
    int nx1, int nx2, int ny1, int ny2, float **vx, float **vy, float **ux,
    float **p, float **pi, float **rho, const float *hc, const float *K_x,
    const float *a_x, const float *b_x, const float *K_y, const float *a_y,
    const float *b_y, float **psi_vxx, float **psi_vyy, int mode,
    int fdorder, int invmat1, int fw, int free_surf, int boundary,
    int nprocx, int nprocy, int posx, int posy, int grad_form,
    float dt, float dh)
{
    int nx = nx2 - nx1 + 1, ny = ny2 - ny1 + 1;
    int sx_vx = (int)(vx[ny1 + 1] - vx[ny1]);
    int sx_vy = (int)(vy[ny1 + 1] - vy[ny1]);
    int sx_ux = (int)(ux[ny1 + 1] - ux[ny1]);
    int sx_p = (int)(p[ny1 + 1] - p[ny1]);
    int sx_pi = (int)(pi[ny1 + 1] - pi[ny1]);
    int sx_rho = (int)(rho[ny1 + 1] - rho[ny1]);
    int sx_psix = (int)(psi_vxx[ny1 + 1] - psi_vxx[ny1]);
    int sx_psiy = (int)(psi_vyy[2] - psi_vyy[1]);
    dim3 block(32, 8);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    update_s_acoustic_kernel<<<grid, block>>>(
        nx, ny, sx_vx, sx_vy, sx_ux, sx_p, sx_pi, sx_rho, sx_psix, sx_psiy,
        vx[ny1] + nx1, vy[ny1] + nx1, ux[ny1] + nx1, p[ny1] + nx1,
        pi[ny1] + nx1, rho[ny1] + nx1, hc, K_x, a_x, b_x, K_y, a_y, b_y,
        psi_vxx[ny1] + 1, psi_vyy[1] + nx1, mode, fdorder / 2, invmat1, fw,
        free_surf, boundary, nprocx, nprocy, posx, posy, grad_form, dt, dh);
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "DENISE GPU acoustic stress update failed: %s\n",
                cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

__global__ static void update_v_acoustic_kernel(
    int nx, int ny, int sx_vx, int sx_vy, int sx_vxp1, int sx_vyp1,
    int sx_p, int sx_rip, int sx_rjp, int sx_psix, int sx_psiy,
    float *vx, float *vy, float *vxp1, float *vyp1, const float *p,
    const float *rip, const float *rjp, const float *hc,
    const float *K_x_half, const float *a_x_half, const float *b_x_half,
    const float *K_y_half, const float *a_y_half, const float *b_y_half,
    float *psi_p_x, float *psi_p_y, int sw, int fdoh, int fw,
    int free_surf, int boundary, int nprocx, int nprocy, int posx, int posy,
    int grad_form, float dt, float dh)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    float px = 0.0f, py = 0.0f;
    for (int m = 1; m <= fdoh; ++m) {
        px += hc[m] * (p[iy * sx_p + ix + m] -
                       p[iy * sx_p + ix - m + 1]);
        py += hc[m] * (p[(iy + m) * sx_p + ix] -
                       p[(iy - m + 1) * sx_p + ix]);
    }
    int i = ix + 1, j = iy + 1;
    if (!boundary && posx == 0 && i <= fw) {
        float q = b_x_half[i] * psi_p_x[iy * sx_psix + ix] + a_x_half[i] * px;
        psi_p_x[iy * sx_psix + ix] = q;
        px = px / K_x_half[i] + q;
    }
    if (!boundary && posx == nprocx - 1 && i >= nx - fw + 1) {
        int h = i - nx + 2 * fw, qx = h - 1;
        float q = b_x_half[h] * psi_p_x[iy * sx_psix + qx] + a_x_half[h] * px;
        psi_p_x[iy * sx_psix + qx] = q;
        px = px / K_x_half[h] + q;
    }
    if (posy == 0 && !free_surf && j <= fw) {
        float q = b_y_half[j] * psi_p_y[iy * sx_psiy + ix] + a_y_half[j] * py;
        psi_p_y[iy * sx_psiy + ix] = q;
        py = py / K_y_half[j] + q;
    }
    if (posy == nprocy - 1 && j >= ny - fw + 1) {
        int h = j - ny + 2 * fw, qy = h - 1;
        float q = b_y_half[h] * psi_p_y[qy * sx_psiy + ix] + a_y_half[h] * py;
        psi_p_y[qy * sx_psiy + ix] = q;
        py = py / K_y_half[h] + q;
    }
    float ax = rip[iy * sx_rip + ix] * px / dh;
    float ay = rjp[iy * sx_rjp + ix] * py / dh;
    if (grad_form == 1) {
        if (sw == 0) { vxp1[iy * sx_vxp1 + ix] = ax; vyp1[iy * sx_vyp1 + ix] = ay; }
        else { vxp1[iy * sx_vxp1 + ix] += dt * vx[iy * sx_vx + ix];
               vyp1[iy * sx_vyp1 + ix] += dt * vy[iy * sx_vy + ix]; }
    } else if (grad_form == 2) {
        if (sw == 0) { vxp1[iy * sx_vxp1 + ix] = ax; vyp1[iy * sx_vyp1 + ix] = ay; }
        else { vxp1[iy * sx_vxp1 + ix] = vx[iy * sx_vx + ix];
               vyp1[iy * sx_vyp1 + ix] = vy[iy * sx_vy + ix]; }
    }
    vx[iy * sx_vx + ix] += dt * ax;
    vy[iy * sx_vy + ix] += dt * ay;
}

extern "C" int denise_gpu_update_v_acoustic(
    int nx1, int nx2, int ny1, int ny2, float **vx, float **vy,
    float **vxp1, float **vyp1, float **p, float **rip, float **rjp,
    const float *hc, const float *K_x_half, const float *a_x_half,
    const float *b_x_half, const float *K_y_half, const float *a_y_half,
    const float *b_y_half, float **psi_p_x, float **psi_p_y, int sw,
    int fdorder, int fw, int free_surf, int boundary, int nprocx, int nprocy,
    int posx, int posy, int grad_form, float dt, float dh)
{
    int nx = nx2 - nx1 + 1, ny = ny2 - ny1 + 1;
#define STRIDE(a, row) ((int)((a)[(row) + 1] - (a)[(row)]))
    int sx_vx=STRIDE(vx,ny1), sx_vy=STRIDE(vy,ny1);
    int sx_vxp1=STRIDE(vxp1,ny1), sx_vyp1=STRIDE(vyp1,ny1);
    int sx_p=STRIDE(p,ny1), sx_rip=STRIDE(rip,ny1), sx_rjp=STRIDE(rjp,ny1);
    int sx_psix=STRIDE(psi_p_x,ny1), sx_psiy=STRIDE(psi_p_y,1);
#undef STRIDE
    dim3 block(32,8), grid((nx+31)/32,(ny+7)/8);
    update_v_acoustic_kernel<<<grid,block>>>(nx,ny,sx_vx,sx_vy,sx_vxp1,
        sx_vyp1,sx_p,sx_rip,sx_rjp,sx_psix,sx_psiy,vx[ny1]+nx1,
        vy[ny1]+nx1,vxp1[ny1]+nx1,vyp1[ny1]+nx1,p[ny1]+nx1,
        rip[ny1]+nx1,rjp[ny1]+nx1,hc,K_x_half,a_x_half,b_x_half,
        K_y_half,a_y_half,b_y_half,psi_p_x[ny1]+1,psi_p_y[1]+nx1,
        sw,fdorder/2,fw,free_surf,boundary,nprocx,nprocy,posx,posy,
        grad_form,dt,dh);
    cudaError_t status=cudaDeviceSynchronize();
    if(status!=cudaSuccess){fprintf(stderr,"DENISE GPU acoustic velocity update failed: %s\n",cudaGetErrorString(status));return -1;}
    return 0;
}

__global__ static void psource_ac_kernel(int nt, int nt_total, float dt,
    float *p, int sp, const float *srcx, const float *srcy,
    const float *signals, int ss, int nsrc, int sw, int quelltypb)
{
    int l=blockIdx.x*blockDim.x+threadIdx.x;
    if(l>=nsrc) return;
    if(sw==1 && quelltypb<4) return;
    int i=(int)srcx[l], j=(int)srcy[l];
    float amp;
    if(sw==1) amp=signals[l*ss+nt];
    else if(nt==1) amp=signals[l*ss+nt+1]/dt;
    else if(nt<nt_total) amp=(signals[l*ss+nt+1]-signals[l*ss+nt-1])/dt;
    else amp=-signals[l*ss+nt-1]/dt;
    atomicAdd(&p[j*sp+i],amp);
}

extern "C" int denise_gpu_psource_ac(int nt,int nt_total,float dt,float **p,
    float **srcpos,float **signals,int nsrc,int sw,int quelltypb)
{
    if(nsrc<=0) return 0;
    int sp=(int)(p[2]-p[1]), ss=(int)(signals[2]-signals[1]);
    psource_ac_kernel<<<(nsrc+127)/128,128>>>(nt,nt_total,dt,p[0],sp,
        srcpos[1]+1,srcpos[2]+1,signals[1],ss,nsrc,sw,quelltypb);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){fprintf(stderr,"DENISE GPU acoustic source failed: %s\n",cudaGetErrorString(e));return -1;} return 0;
}

__global__ static void surface_ac_kernel(int nx,int fdoh,float *p,int sp)
{
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    if(ix>=nx) return;
    int i=ix+1; p[i]=0.0f;
    for(int m=1;m<=fdoh;++m) p[-m*sp+i]=-p[m*sp+i];
}

extern "C" int denise_gpu_surface_ac(int ndepth,int nx,int fdorder,float **p)
{
    int sp=(int)(p[ndepth+1]-p[ndepth]);
    surface_ac_kernel<<<(nx+127)/128,128>>>(nx,fdorder/2,p[ndepth],sp);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){fprintf(stderr,"DENISE GPU acoustic surface failed: %s\n",cudaGetErrorString(e));return -1;} return 0;
}

__global__ static void seismo_ac_kernel(int sample,int ntr,const int *rx,
 const int *ry,float *svx,int ssvx,float *svy,int ssvy,float *spres,int ssp,
 float *scurl,int ssc,float *sdiv,int ssd,const float *vx,const float *vy,
 const float *p,const float *pi,const float *u,const float *rho,int sw,
 const float *hc,int seismo,int fdoh,int invmat1,float dt,float dh)
{
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=ntr)return;
 int i=rx[t],j=ry[t],out=sample;
 if(seismo==1||seismo==4){svx[t*ssvx+out]=vx[j*sw+i];svy[t*ssvy+out]=vy[j*sw+i];}
 if(seismo==2||seismo==4)spres[t*ssp+out]=p[j*sw+i];
 if(seismo==3||seismo==4){float vxx=0,vyy=0,vyx=0,vxy=0;
  for(int m=1;m<=fdoh;m++){vxx+=hc[m]*(vx[j*sw+i+m-1]-vx[j*sw+i-m]);vyy+=hc[m]*(vy[(j+m-1)*sw+i]-vy[(j-m)*sw+i]);vyx+=hc[m]*(vy[j*sw+i+m]-vy[j*sw+i-m+1]);vxy+=hc[m]*(vx[(j+m)*sw+i]-vx[(j-m+1)*sw+i]);}
  vxx*=dt/dh;vyy*=dt/dh;vyx*=dt/dh;vxy*=dt/dh;
  float mu,la;if(invmat1==1){mu=u[j*sw+i]*u[j*sw+i]*rho[j*sw+i];la=pi[j*sw+i]*pi[j*sw+i]*rho[j*sw+i];}else{mu=u[j*sw+i];la=pi[j*sw+i];}
  sdiv[t*ssd+out]=(vxx+vyy)*sqrtf(la);scurl[t*ssc+out]=(vxy-vyx)*sqrtf(mu);
 }
}

extern "C" int denise_gpu_seismo_ac(int sample,int ntr,int **recpos,
 float **sectionvx,float **sectionvy,float **sectionp,float **sectioncurl,
 float **sectiondiv,float **vx,float **vy,float **p,float **pi,float **u,
 float **rho,const float *hc,int seismo,int fdorder,int invmat1,float dt,float dh)
{
 if(ntr<=0)return 0;
#define BASE(a) ((a)?(a)[1]:NULL)
#define SSEC(a) ((a)?(int)((a)[2]-(a)[1]):0)
 int sw=(int)(vx[1]-vx[0]);
 seismo_ac_kernel<<<(ntr+127)/128,128>>>(sample,ntr,recpos[1]+1,recpos[2]+1,
 BASE(sectionvx),SSEC(sectionvx),BASE(sectionvy),SSEC(sectionvy),BASE(sectionp),
 SSEC(sectionp),BASE(sectioncurl),SSEC(sectioncurl),BASE(sectiondiv),SSEC(sectiondiv),
 vx[0],vy[0],p[0],pi[0],u[0],rho[0],sw,hc,seismo,fdorder/2,invmat1,dt,dh);
#undef BASE
#undef SSEC
 cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess){fprintf(stderr,"DENISE GPU acoustic receiver failed: %s\n",cudaGetErrorString(e));return -1;}return 0;
}
