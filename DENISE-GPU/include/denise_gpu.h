#ifndef DENISE_GPU_H
#define DENISE_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

int denise_gpu_init(int world_rank);
void denise_gpu_finalize(void);
int denise_gpu_sync(const char *where);
void *denise_gpu_managed_alloc(unsigned long bytes);
void denise_gpu_managed_free(void *ptr);
int denise_gpu_prefetch_all(void);

int denise_gpu_update_s_acoustic(
    int nx1, int nx2, int ny1, int ny2, float **vx, float **vy, float **ux,
    float **p, float **pi, float **rho, const float *hc, const float *K_x,
    const float *a_x, const float *b_x, const float *K_y, const float *a_y,
    const float *b_y, float **psi_vxx, float **psi_vyy, int mode,
    int fdorder, int invmat1, int fw, int free_surf, int boundary,
    int nprocx, int nprocy, int posx, int posy, int grad_form,
    float dt, float dh);

int denise_gpu_update_v_acoustic(
    int nx1, int nx2, int ny1, int ny2, float **vx, float **vy,
    float **vxp1, float **vyp1, float **p, float **rip, float **rjp,
    const float *hc, const float *K_x_half, const float *a_x_half,
    const float *b_x_half, const float *K_y_half, const float *a_y_half,
    const float *b_y_half, float **psi_p_x, float **psi_p_y, int sw,
    int fdorder, int fw, int free_surf, int boundary, int nprocx, int nprocy,
    int posx, int posy, int grad_form, float dt, float dh);
int denise_gpu_psource_ac(int nt, int nt_total, float dt, float **p,
                          float **srcpos, float **signals, int nsrc,
                          int sw, int quelltypb);
int denise_gpu_surface_ac(int ndepth, int nx, int fdorder, float **p);
int denise_gpu_seismo_ac(int sample,int ntr,int **recpos,float **sectionvx,
 float **sectionvy,float **sectionp,float **sectioncurl,float **sectiondiv,
 float **vx,float **vy,float **p,float **pi,float **u,float **rho,
 const float *hc,int seismo,int fdorder,int invmat1,float dt,float dh);
int denise_gpu_update_s_psv(int nx1,int nx2,int ny1,int ny2,float **vx,
 float **vy,float **ux,float **uy,float **uxy,float **sxx,float **syy,
 float **sxy,float **pi,float **u,float **uipjp,float **rho,const float *hc,
 const float *K_x,const float *a_x,const float *b_x,const float *K_x_half,
 const float *a_x_half,const float *b_x_half,const float *K_y,const float *a_y,
 const float *b_y,const float *K_y_half,const float *a_y_half,
 const float *b_y_half,float **psi_vxx,float **psi_vyy,float **psi_vxy,
 float **psi_vyx,int mode,int fdorder,int invmat1,int fw,int free_surf,
 int boundary,int nprocx,int nprocy,int posx,int posy,int grad_form,
 float dt,float dh);
int denise_gpu_update_v_psv(int nx1,int nx2,int ny1,int ny2,float **vx,
 float **vy,float **vxp1,float **vyp1,float **p,float **q,float **t,
 float **rip,float **rjp,const float *hc,const float *Kx,const float *ax,
 const float *bx,const float *Kxh,const float *axh,const float *bxh,
 const float *Ky,const float *ay,const float *by,const float *Kyh,
 const float *ayh,const float *byh,float **psxx,float **psyy,float **psxyy,
 float **psxyx,int sw,int fdorder,int fw,int free_surf,int boundary,
 int nprocx,int nprocy,int posx,int posy,int grad,float dt,float dh);
int denise_gpu_psource_psv(int nt,int nt_total,float dt,float **sxx,float **syy,
 float **srcpos,float **signals,int nsrc,int sw,int quelltypb);
int denise_gpu_vsource_psv(int nt,float **vx,float **vy,float **srcpos,
 float **signals_x,float **signals_y,int nsrc,int sw,int quelltypb);
int denise_gpu_surface_psv(int depth,int nx,int fdorder,int invmat,int fw,
 int boundary,int nprocx,int posx,float dt,float dh,float **vx,float **vy,
 float **sxx,float **syy,float **sxy,float **pi,float **u,float **rho,
 const float *hc,const float *Kx,const float *ax,const float *bx,float **psi);
int denise_gpu_seismo_psv(int sample,int ntr,int **recpos,float **sectionvx,
 float **sectionvy,float **sectionp,float **sectioncurl,float **sectiondiv,
 float **vx,float **vy,float **sxx,float **syy,float **pi,float **u,float **rho,
 const float *hc,int seismo,int fdorder,int invmat,float dt,float dh);
int denise_gpu_update_s_sh(int nx1,int nx2,int ny1,int ny2,float **vz,float **uz,
 float **uzx,float **syz,float **sxz,float **ujp,float **uip,const float *hc,
 const float *Kxh,const float *axh,const float *bxh,const float *Kyh,
 const float *ayh,const float *byh,float **pvzx,float **pvzy,int mode,int fdorder,
 int fw,int free_surf,int boundary,int nprocx,int nprocy,int posx,int posy,
 int grad,float dt,float dh);
int denise_gpu_update_v_sh(int nx1,int nx2,int ny1,int ny2,float **vz,
 float **vzp,float **sxz,float **syz,float **rhoi,const float *hc,
 const float *Kx,const float *ax,const float *bx,const float *Ky,const float *ay,
 const float *by,float **psxz,float **psyz,int sw,int fdorder,int fw,int free_surf,
 int boundary,int nprocx,int nprocy,int posx,int posy,int grad,float dt,float dh);
int denise_gpu_update_s_aniso(int nx1,int nx2,int ny1,int ny2,float **vx,
 float **vy,float **ux,float **uy,float **uxy,float **sxx,float **syy,float **sxy,
 float **a11,float **a13,float **a15,float **a15h,float **a33,float **a35,
 float **a35h,float **a55,
 const float *hc,const float *Kx,const float *ax,const float *bx,const float *Kxh,
 const float *axh,const float *bxh,const float *Ky,const float *ay,const float *by,
 const float *Kyh,const float *ayh,const float *byh,float **pvxx,float **pvyy,
 float **pvxy,float **pvyx,int mode,int fdorder,int fw,int free_surf,int boundary,
 int nprocx,int nprocy,int posx,int posy,int grad,float dt,float dh,int tilted);
int denise_gpu_update_s_visc_sh(int nx1,int nx2,int ny1,int ny2,float **vz,
 float **uz,float **uzx,float **syz,float **sxz,float ***r,float ***q,
 float **fipjp,float **f,float *bip,float *bjm,float *cip,float *cjm,
 float ***d,float ***dip,const float *hc,const float *Kxh,const float *axh,
 const float *bxh,const float *Ky,const float *ay,const float *by,
 const float *Kyh,const float *ayh,const float *byh,float **pvzx,float **pvzy,
 int mode,int fdorder,int fw,int free_surf,int boundary,int nprocx,int nprocy,
 int posx,int posy,int grad,float dt,float dh,int mechanisms);
int denise_gpu_update_s_visc_psv(int nx1,int nx2,int ny1,int ny2,float **vx,
 float **vy,float **ux,float **uy,float **uxy,float **sxx,float **syy,float **sxy,
 float ***r,float ***p,float ***q,float **fipjp,float **f,float **g,float *bip,
 float *bjm,float *cip,float *cjm,float ***d,float ***e,float ***dip,
 const float *hc,const float *Kx,const float *ax,const float *bx,const float *Kxh,
 const float *axh,const float *bxh,const float *Ky,const float *ay,const float *by,
 const float *Kyh,const float *ayh,const float *byh,float **pvxx,float **pvyy,
 float **pvxy,float **pvyx,int mode,int fdorder,int fw,int free_surf,int boundary,
 int nprocx,int nprocy,int posx,int posy,int grad,float dt,float dh,int mechanisms);
int denise_gpu_fwi_store_ac(float **vx,float **vy,float **p,float **ux,
 float *frx,float *fry,float *fp,int nx,int ny,int idx,int idy,int offset,int grad);
int denise_gpu_fwi_corr_ac(float **vxp,float **vyp,float **p,float **gl,float **gr,
 const float *frx,const float *fry,const float *fp,int nx,int ny,int idx,int idy,
 int offset);
int denise_gpu_fwi_store_sh(float **vz,float **sxz,float **syz,float **uzx,float **uz,
 float *fr,float *fx,float *fy,int nx,int ny,int idx,int idy,int offset,int grad);
int denise_gpu_fwi_corr_sh(float **vzp,float **sxz,float **syz,float **gr,float **gu,
 const float *fr,const float *fx,const float *fy,float **rho,float **mu,int nx,int ny,
 int idx,int idy,int offset,int invmat);
int denise_gpu_fwi_store_psv(float **vx,float **vy,float **sxx,float **syy,float **sxy,
 float **ux,float **uy,float **uxy,float *frx,float *fry,float *fx,float *fy,float *fu,
 int nx,int ny,int idx,int idy,int offset,int grad);
int denise_gpu_fwi_corr_psv(float **vx,float **vy,float **sxx,float **syy,float **sxy,
 float **gl,float **gm,float **gr,const float *frx,const float *fry,const float *fx,
 const float *fy,const float *fu,float **rho,float **vp,float **vs,int nx,int ny,
 int idx,int idy,int offset,int grad,int invmat);
int denise_gpu_fwi_corr_aniso(float **vx,float **vy,float **sxx,float **syy,
 float **gl,float **gr,const float *frx,const float *fry,const float *fx,
 const float *fy,int nx,int ny,int idx,int idy,int offset);
int denise_gpu_eprecond(float **energy,float **vx,float **vy,
 int nx,int ny,int idx,int idy);

#ifdef __cplusplus
}
#endif

#endif
