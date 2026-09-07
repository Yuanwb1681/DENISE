#ifndef DENISE_SINGLE_GPU_MPI_SHIM_H
#define DENISE_SINGLE_GPU_MPI_SHIM_H

/* Single-process compatibility layer.  It intentionally implements only the
 * MPI subset used by DENISE; no MPI library is linked in the standalone build. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int MPI_Comm;
typedef int MPI_Request;
typedef int MPI_Status;
typedef int MPI_Datatype;
typedef int MPI_Op;

#define MPI_COMM_WORLD 0
#define MPI_FLOAT 1
#define MPI_DOUBLE 2
#define MPI_INT 3
#define MPI_SUM 1
#define MPI_MAX 2
#define MPI_MIN 3
#define MPI_SUCCESS 0
#define MPI_STATUS_IGNORE ((MPI_Status *)0)

static inline size_t denise_mpi_type_size(MPI_Datatype type) {
    return type == MPI_DOUBLE ? sizeof(double) : type == MPI_INT ? sizeof(int) : sizeof(float);
}
static inline int MPI_Init(int *argc,char ***argv){(void)argc;(void)argv;return 0;}
static inline int MPI_Finalize(void){return 0;}
static inline int MPI_Abort(MPI_Comm c,int code){(void)c;exit(code);return code;}
static inline int MPI_Comm_rank(MPI_Comm c,int *rank){(void)c;*rank=0;return 0;}
static inline int MPI_Comm_size(MPI_Comm c,int *size){(void)c;*size=1;return 0;}
static inline int MPI_Comm_split(MPI_Comm c,int color,int key,MPI_Comm *out){(void)c;(void)color;(void)key;*out=0;return 0;}
static inline int MPI_Comm_free(MPI_Comm *c){*c=0;return 0;}
static inline int MPI_Barrier(MPI_Comm c){(void)c;return 0;}
static inline int MPI_Bcast(void *p,int n,MPI_Datatype t,int root,MPI_Comm c){(void)p;(void)n;(void)t;(void)root;(void)c;return 0;}
static inline int MPI_Allreduce(const void *s,void *r,int n,MPI_Datatype t,MPI_Op op,MPI_Comm c){(void)op;(void)c;if(s!=r)memcpy(r,s,(size_t)n*denise_mpi_type_size(t));return 0;}
static inline int MPI_Buffer_attach(void *p,int n){(void)p;(void)n;return 0;}
static inline int MPI_Buffer_detach(void *p,int *n){(void)p;if(n)*n=0;return 0;}
static inline int MPI_Bsend(const void *p,int n,MPI_Datatype t,int d,int tag,MPI_Comm c){(void)p;(void)n;(void)t;(void)d;(void)tag;(void)c;return 0;}
static inline int MPI_Recv(void *p,int n,MPI_Datatype t,int s,int tag,MPI_Comm c,MPI_Status *st){(void)p;(void)n;(void)t;(void)s;(void)tag;(void)c;(void)st;return 0;}
static inline int MPI_Sendrecv_replace(void *p,int n,MPI_Datatype t,int d,int stag,int s,int rtag,MPI_Comm c,MPI_Status *st){(void)p;(void)n;(void)t;(void)d;(void)stag;(void)s;(void)rtag;(void)c;(void)st;return 0;}
static inline int MPI_Bsend_init(const void *p,int n,MPI_Datatype t,int d,int tag,MPI_Comm c,MPI_Request *r){(void)p;(void)n;(void)t;(void)d;(void)tag;(void)c;*r=0;return 0;}
static inline int MPI_Recv_init(void *p,int n,MPI_Datatype t,int s,int tag,MPI_Comm c,MPI_Request *r){(void)p;(void)n;(void)t;(void)s;(void)tag;(void)c;*r=0;return 0;}
static inline int MPI_Start(MPI_Request *r){(void)r;return 0;}
static inline int MPI_Wait(MPI_Request *r,MPI_Status *s){(void)r;(void)s;return 0;}
static inline double MPI_Wtime(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+1e-9*(double)t.tv_nsec;}

#endif
