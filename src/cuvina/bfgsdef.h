#ifndef BFGSDEF_H
#define BFGSDEF_H

#define MAX_TRIALS 8
// make sure X >= 4 && X < Y
#define NRTHREADS 256
#define DIMX 4
#define DIMY 16
#define DIMXY ((DIMX)*(DIMY))
#define DIMZ ((NRTHREADS)/(DIMXY))
#define WARPSZ ((NRTHREADS)/32)
#if MAX_TRIALS % DIMZ != 0
#error MAX_TRIALS must be multiples of DIMZ to get a better performance
#endif

#if DIMXY > 32
#define XYSYNC() SYNC()
#else
#define XYSYNC() WARPSYNC()
#endif

#if EVAL_IN_WARP
// #if DIMX * DIMY != 32
// #error to eval model der in a single warp, block dim x * y must be 32 to avoid sync
// #endif
#endif
#endif