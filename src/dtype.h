#ifndef DTYPE_H
#define DTYPE_H

#ifndef USE_DOUBLE
#error must define USE_DOUBLE
#endif

#if USE_DOUBLE
using dtype = double;
#else
using dtype = float;
#endif

#endif