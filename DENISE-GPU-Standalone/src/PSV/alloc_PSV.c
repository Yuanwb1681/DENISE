/*
 * Allocate memory for PSV problem 
 *
 * Daniel Koehn
 * Kiel, 21/01/2016
 */

#include "fd.h"

void alloc_PSV(struct wavePSV *wavePSV, struct wavePSV_PML *wavePSV_PML){

        /* global variables */
	extern int NX, NY, L, FW, FDORDER;

	/* local variables */
	int nd;

        nd = FDORDER/2 + 1;	

	/* Live elastic fields are never dereferenced by the host in the standalone
	 * no-snapshot path. Give them stable device-only storage. */
#ifdef DENISE_STANDALONE_GPU
#define WAVE_MATRIX device_matrix
#else
#define WAVE_MATRIX matrix
#endif
        (*wavePSV).psxx =  WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd);
	(*wavePSV).psxy = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd); (*wavePSV).psyy = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd);
	(*wavePSV).pvx = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd); (*wavePSV).pvy = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd);
	(*wavePSV).pvxp1 = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd); (*wavePSV).pvyp1 = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd);
	(*wavePSV).pvxm1 = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd); (*wavePSV).pvym1 = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd);
	(*wavePSV).ux = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd); (*wavePSV).uy = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd);
	(*wavePSV).uxy = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd); (*wavePSV).uyx = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd);
	(*wavePSV).uttx = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd); (*wavePSV).utty = WAVE_MATRIX(-nd+1,NY+nd,-nd+1,NX+nd);
#undef WAVE_MATRIX

        /* memory allocation for visco-elastic wavefield variables */
	if (L > 0) {
	    (*wavePSV).pr = f3tensor(-nd+1,NY+nd,-nd+1,NX+nd,1,L);
	    (*wavePSV).pp = f3tensor(-nd+1,NY+nd,-nd+1,NX+nd,1,L);
	    (*wavePSV).pq = f3tensor(-nd+1,NY+nd,-nd+1,NX+nd,1,L);
	}

        /* memory allocation for PML variables */
        if(FW>0){
#ifdef DENISE_STANDALONE_GPU
#define PML_MATRIX device_matrix
#else
#define PML_MATRIX matrix
#endif

	  (*wavePSV_PML).d_x = vector(1,2*FW);
	  (*wavePSV_PML).K_x = vector(1,2*FW);
	  (*wavePSV_PML).alpha_prime_x = vector(1,2*FW);
	  (*wavePSV_PML).a_x = vector(1,2*FW);
	  (*wavePSV_PML).b_x = vector(1,2*FW);
	  
	  (*wavePSV_PML).d_x_half = vector(1,2*FW);
	  (*wavePSV_PML).K_x_half = vector(1,2*FW);
	  (*wavePSV_PML).alpha_prime_x_half = vector(1,2*FW);
	  (*wavePSV_PML).a_x_half = vector(1,2*FW);
	  (*wavePSV_PML).b_x_half = vector(1,2*FW);

	  (*wavePSV_PML).d_y = vector(1,2*FW);
	  (*wavePSV_PML).K_y = vector(1,2*FW);
	  (*wavePSV_PML).alpha_prime_y = vector(1,2*FW);
	  (*wavePSV_PML).a_y = vector(1,2*FW);
	  (*wavePSV_PML).b_y = vector(1,2*FW);
	  
	  (*wavePSV_PML).d_y_half = vector(1,2*FW);
	  (*wavePSV_PML).K_y_half = vector(1,2*FW);
	  (*wavePSV_PML).alpha_prime_y_half = vector(1,2*FW);
	  (*wavePSV_PML).a_y_half = vector(1,2*FW);
	  (*wavePSV_PML).b_y_half = vector(1,2*FW);

	  (*wavePSV_PML).psi_sxx_x = PML_MATRIX(1,NY,1,2*FW);
	  (*wavePSV_PML).psi_syy_y = PML_MATRIX(1,2*FW,1,NX);
	  (*wavePSV_PML).psi_sxy_y = PML_MATRIX(1,2*FW,1,NX);
	  (*wavePSV_PML).psi_sxy_x = PML_MATRIX(1,NY,1,2*FW);
	  (*wavePSV_PML).psi_vxx = PML_MATRIX(1,NY,1,2*FW);
	  (*wavePSV_PML).psi_vxxs = PML_MATRIX(1,NY,1,2*FW);
	  (*wavePSV_PML).psi_vyy = PML_MATRIX(1,2*FW,1,NX);
	  (*wavePSV_PML).psi_vxy = PML_MATRIX(1,2*FW,1,NX);
	  (*wavePSV_PML).psi_vyx = PML_MATRIX(1,NY,1,2*FW);
#undef PML_MATRIX
   
        }
	
}

