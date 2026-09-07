/*------------------------------------------------------------------------
 *   stress free surface condition
 *   
 *  D. Koehn,
 *  Kiel, 10.06.2017
 *  ----------------------------------------------------------------------*/

#include "fd.h"
#ifdef DENISE_USE_CUDA
#include "denise_gpu.h"
#endif

void surface_acoustic_PML_AC(int ndepth, float ** p){


	int i,j,m;
	int fdoh;
	extern int NX, FDORDER;
	
	fdoh = FDORDER/2;

#ifdef DENISE_USE_CUDA
	if (!getenv("DENISE_GPU_DISABLE_AC_SURFACE")) {
	if(denise_gpu_surface_ac(ndepth,NX,FDORDER,p)!=0)
		err("CUDA acoustic free surface failed");
	return;
	}
#endif

	j=ndepth;     /* The free surface is located exactly in y=1/2*dh !! */
        for (i=1;i<=NX;i++){
        p[j][i] = 0;
		for (m=1; m<=fdoh; m++) {
			p[j-m][i] = -p[j+m][i];
		}
	}

}
