#ifndef DTYPE_H
#define DTYPE_H

#define USING_DOUBLE 1


#if USING_DOUBLE
using dtype = double;
#else
using dtype = float;
#endif

#endif