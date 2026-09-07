/*
 * Apply time-domain filter
 *
 * Daniel Koehn
 * Kiel, 23/04/2016
 */

#include "fd.h"
#ifdef DENISE_USE_CUDA
#include "denise_gpu.h"
#endif

void apply_tdfilt(float **section, int ntr, int ns, int order, float fc2, float fc1){

     /* global variables */

     /* local variables */

     extern float DT;
#ifdef DENISE_USE_CUDA
     /* Recursive samples remain serial within a trace, while traces execute in
      * parallel. Tiny source arrays stay on the CPU to avoid launch overhead. */
     if(ntr>=32 && !getenv("DENISE_GPU_DISABLE_FILTER") &&
        denise_gpu_filter_traces(section,ntr,ns,order,fc2,DT,0)==0){
       if(fc1<=0.0f || denise_gpu_filter_traces(section,ntr,ns,order,fc1,DT,1)==0)
         return;
     }
#endif
     timedomain_filt(section,fc2,order,ntr,ns,1);

     if(fc1>0.0){ /* band-pass */
       timedomain_filt(section,fc1,order,ntr,ns,2);
     } 
       	
}


