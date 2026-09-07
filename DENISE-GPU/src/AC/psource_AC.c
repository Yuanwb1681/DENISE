/*------------------------------------------------------------------------
 *   generate P-wave source at source nodes
 *   last update 11/06/2017, D. Koehn
 *
 *  ----------------------------------------------------------------------*/

#include "fd.h"
#ifdef DENISE_USE_CUDA
#include "denise_gpu.h"
#endif

void psource_AC(int nt, float ** p,
float **  srcpos_loc, float ** signals, int nsrc, int sw){


        extern int MYID, NT, QUELLTYPB;
	extern float DH, DT;
	int i, j, l;
	float amp;

#ifdef DENISE_USE_CUDA
	if (!getenv("DENISE_GPU_DISABLE_AC_SOURCE")) {
	if(denise_gpu_psource_ac(nt,NT,DT,p,srcpos_loc,signals,nsrc,sw,QUELLTYPB)!=0)
		err("CUDA acoustic source injection failed");
	return;
	}
#endif



	/* adding source wavelet to stress components 
	   (explosive source) at source points */

       if(sw==0){
	   for (l=1;l<=nsrc;l++) {
	   	   i=(int)srcpos_loc[1][l];
		   j=(int)srcpos_loc[2][l];
		
		   if(nt==1){amp=signals[l][nt+1]/DT;}
		   if((nt>1)&&(nt<NT)){amp=(signals[l][nt+1]-signals[l][nt-1])/DT;}
		   if(nt==NT){amp=-signals[l][nt-1]/DT;}
		
		   /*amp = signals[l][nt];*/
		
		   p[j][i]+=amp;

	   }
       }

       if(sw==1)
       {
         if(QUELLTYPB>=4)
         {
           for (l=1;l<=nsrc;l++) {
                i=(int)srcpos_loc[1][l];
                j=(int)srcpos_loc[2][l];
                p[j][i] += signals[l][nt];
           }
         }
       }

}
