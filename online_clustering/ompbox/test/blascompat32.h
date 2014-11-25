/*
 * @(#)blascompat32.h    generated by: makeheader 5.1.5  Mon Jun 28 17:10:48 2010
 *
 *		built from:	../../src/include/copyright.h
 *				../../src/include/pragma_interface.h
 *				include/blascompat32/blascompat32.h
 */

#if defined(_MSC_VER)
//# pragma once
#endif
#if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ > 3))
//# pragma once
#endif

#ifndef blascompat32_h
#define blascompat32_h


/*
 * Copyright 1984-2003 The MathWorks, Inc.
 * All Rights Reserved.
 */



/* Copyright 2003-2006 The MathWorks, Inc. */

/* Only define EXTERN_C if it hasn't been defined already. This allows
 * individual modules to have more control over managing their exports.
 */
#ifndef EXTERN_C

#ifdef __cplusplus
  #define EXTERN_C extern "C"
#else
  #define EXTERN_C extern
#endif

#endif


/*
 * CONFIDENTIAL AND CONTAINING PROPRIETARY TRADE SECRETS
 * Copyright 1984-2009 The MathWorks, Inc.
 * The source code contained in this listing contains proprietary and
 * confidential trade secrets of The MathWorks, Inc.   The use, modification,
 * or development of derivative work based on the code or ideas obtained
 * from the code is prohibited without the express written permission of The
 * MathWorks, Inc.  The disclosure of this code to any party not authorized
 * by The MathWorks, Inc. is strictly forbidden.
 * CONFIDENTIAL AND CONTAINING PROPRIETARY TRADE SECRETS
 */

/*
   32-bit API wrapper for libmwblas
   WARNING: This module is a temporary module specifically designed to 
            bridge the incompatiblity between Embedded MATLAB (which lacks
            a 64-bit integer type) and 64-bit BLAS only.
            Do not link to this module otherwise.
            Link to libmwblas instead.
 */   
#include "tmwtypes.h"

#if defined(_WIN32) 
#define FORTRAN_WRAPPER(x) x
#else
#define FORTRAN_WRAPPER(x) x ## _
#endif

#ifndef COMPLEX_TYPES
#define COMPLEX_TYPES
  typedef struct{float r,i;} complex;
  typedef struct{double r,i;} doublecomplex;
#endif
  
 
#define isamax32 FORTRAN_WRAPPER(isamax32)
EXTERN_C int isamax32(const int *n32, const float  *sx, const int *incx32);


 
#define idamax32 FORTRAN_WRAPPER(idamax32)
EXTERN_C int idamax32(const int *n32, const double *dx, const int *incx32);


 
#define icamax32 FORTRAN_WRAPPER(icamax32)
EXTERN_C int icamax32(const int *n32, const creal32_T *cx, const int *incx32);


 
#define izamax32 FORTRAN_WRAPPER(izamax32)
EXTERN_C int izamax32(const int *n32, const creal_T *zx, const int *incx32);


 
#define sasum32 FORTRAN_WRAPPER(sasum32)
EXTERN_C float sasum32(const int *n32, const float  *sx, const int *incx32);


 
#define dasum32 FORTRAN_WRAPPER(dasum32)
EXTERN_C double dasum32(const int *n32, const double *dx, const int *incx32);



#define saxpy32 FORTRAN_WRAPPER(saxpy32)
EXTERN_C void saxpy32(const int *n32, const float *sa, const float  *sx,
             const int *incx32, float *sy, const int *incy32);



#define daxpy32 FORTRAN_WRAPPER(daxpy32)
EXTERN_C void daxpy32(const int *n32, const double *da, const double *dx,
             const int *incx32, double *dy, const int *incy32);



#define caxpy32 FORTRAN_WRAPPER(caxpy32)
EXTERN_C void caxpy32(const int *n32, const creal32_T *ca, const creal32_T *cx,
             const int *incx32, creal32_T *cy, const int *incy32);



#define zaxpy32 FORTRAN_WRAPPER(zaxpy32)
EXTERN_C void zaxpy32(const int *n32, const creal_T *za, const creal_T *zx,
             const int *incx32, creal_T *zy, const int *incy32);



#define scopy32 FORTRAN_WRAPPER(scopy32)
EXTERN_C void scopy32(const int *n32, const float *sx, const int *incx32, 
             float *sy, const int *incy32);



#define dcopy32 FORTRAN_WRAPPER(dcopy32)
EXTERN_C void dcopy32(const int *n32, const double *dx, const int *incx32, 
             double *dy, const int *incy32);



#define ccopy32 FORTRAN_WRAPPER(ccopy32)
EXTERN_C void ccopy32(const int *n32, const creal32_T *cx, const int *incx32, 
             creal32_T *cy, const int *incy32);



#define zcopy32 FORTRAN_WRAPPER(zcopy32)
EXTERN_C void zcopy32(const int *n32, creal_T *zx, const int *incx32, 
             creal_T *zy, const int *incy32);



#define cdotc32 FORTRAN_WRAPPER(cdotc32)
EXTERN_C complex cdotc32(const int *n32, const creal32_T *cx, const int *incx32,
              const creal32_T *cy, const int *incy32);



#define zdotc32 FORTRAN_WRAPPER(zdotc32)
EXTERN_C doublecomplex zdotc32(const int *n32, const creal_T *zx, const int *incx32,
               const creal_T *zy, const int *incy32);



#define sdot32 FORTRAN_WRAPPER(sdot32)
EXTERN_C float sdot32(const int *n32, const float *sx, const int *incx32, 
             const float  *sy, const int *incy32);



#define ddot32 FORTRAN_WRAPPER(ddot32)
EXTERN_C double ddot32(const int *n32, const double *dx, const int *incx32, 
              const double *dy, const int *incy32);



#define cdotu32 FORTRAN_WRAPPER(cdotu32)
EXTERN_C complex cdotu32(const int *n32, const creal32_T *cx, const int *incx32, 
              const creal32_T *cy, const int *incy32);



#define zdotu32 FORTRAN_WRAPPER(zdotu32)
EXTERN_C doublecomplex zdotu32(const int *n32, const creal_T *zx, const int *incx32, 
               const creal_T *zy, const int *incy32);



#define sgemm32 FORTRAN_WRAPPER(sgemm32)
EXTERN_C void sgemm32(char *transa, char *transb, const int *m32, const int *n32,
             const int *k32, const float *alpha, const float *a, const int *lda32,
             const float *b, const int *ldb32, const float *beta, float *c,
             const int *ldc32);



#define dgemm32 FORTRAN_WRAPPER(dgemm32)
EXTERN_C void dgemm32(char *transa, char *transb, const int *m32, const int *n32,
             const int *k32, const double *alpha, const double *a, const int *lda32,
             const double *b, const int *ldb32, const double *beta, double  *c,
             const int *ldc32);



#define cgemm32 FORTRAN_WRAPPER(cgemm32)
EXTERN_C void cgemm32(char *transa, char *transb, const int *m32, const int *n32,
             const int *k32, const creal32_T *alpha, const creal32_T *a, const int *lda32,
             const creal32_T *b, const int *ldb32, const creal32_T *beta, creal32_T *c,
             const int *ldc32);



#define zgemm32 FORTRAN_WRAPPER(zgemm32)
EXTERN_C void zgemm32(char *transa, char *transb, const int *m32, const int *n32,
             const int *k32, const creal_T *alpha, const creal_T *a, const int *lda32,
             const creal_T *b, const int *ldb32, const creal_T *beta, creal_T *c,
             const int *ldc32);



#define sgemv32 FORTRAN_WRAPPER(sgemv32)
EXTERN_C void sgemv32(char *trans, const int *m32, const int *n32, const float *alpha,
             const float *a, const int *lda32, const float *x, const int *incx32, 
             const float *beta, float *y, const int *incy32);



#define dgemv32 FORTRAN_WRAPPER(dgemv32)
EXTERN_C void dgemv32(char *trans, const int *m32, const int *n32, const double *alpha,
             const double *a, const int *lda32, const double *x, const int *incx32, 
             const double *beta, double *y, const int *incy32);



#define cgemv32 FORTRAN_WRAPPER(cgemv32)
EXTERN_C void cgemv32(char *trans, const int *m32, const int *n32, const creal32_T *alpha,
             const creal32_T *a, const int *lda32, const creal32_T *x, const int *incx32, 
             const creal32_T *beta, creal32_T *y, const int *incy32);



#define zgemv32 FORTRAN_WRAPPER(zgemv32)
EXTERN_C void zgemv32(char *trans, const int *m32, const int *n32, const creal_T *alpha,
             const creal_T *a, const int *lda32, const creal_T *x, const int *incx32, 
             const creal_T *beta, creal_T *y, const int *incy32);



#define cgerc32 FORTRAN_WRAPPER(cgerc32)
EXTERN_C void cgerc32(const int *m32, const int *n32, const creal32_T *alpha, const creal32_T *x,
             const int *incx32, const creal32_T *y, const int *incy32,
             creal32_T *a, const int *lda32);

    


#define zgerc32 FORTRAN_WRAPPER(zgerc32)
EXTERN_C void zgerc32(const int *m32, const int *n32, const creal_T *alpha, const creal_T *x,
             const int *incx32, creal_T *y, const int *incy32,
             creal_T *a, const int *lda32);

     

#define sger32 FORTRAN_WRAPPER(sger32)
EXTERN_C void sger32(const int *m32, const int *n32, const float *alpha, const float  *x,
            const int *incx32, const float *y, const int *incy32, 
            float *a, const int *lda32);



#define dger32 FORTRAN_WRAPPER(dger32)
EXTERN_C void dger32(const int *m32, const int *n32, const double *alpha, const double *x,
            const int *incx32, const double *y, const int *incy32,
            double *a, const int *lda32);



#define cgeru32 FORTRAN_WRAPPER(cgeru32)
EXTERN_C void cgeru32(const int *m32, const int *n32, const creal32_T *alpha, const creal32_T *x,
             const int *incx32, const creal32_T *y, const int *incy32,
             creal32_T *a, const int *lda32);



#define zgeru32 FORTRAN_WRAPPER(zgeru32)
EXTERN_C void zgeru32(const int *m32, const int *n32, const creal_T *alpha, const creal_T *x,
             const int *incx32, creal_T *y, const int *incy32,
             creal_T *a, const int *lda32);



#define snrm232 FORTRAN_WRAPPER(snrm232)
EXTERN_C float snrm232(const int *n32, const float *x, const int *incx32);



#define dnrm232 FORTRAN_WRAPPER(dnrm232)
EXTERN_C double dnrm232(const int *n32, const double *x, const int *incx32);



#define scnrm232 FORTRAN_WRAPPER(scnrm232)
EXTERN_C float scnrm232(const int *n32, const creal32_T *x, const int *incx32);



#define dznrm232 FORTRAN_WRAPPER(dznrm232)
EXTERN_C double dznrm232(const int *n32, const creal_T *x, const int *incx32);



#define srotg32 FORTRAN_WRAPPER(srotg32)
EXTERN_C void srotg32(float  *sa, float  *sb, float  *c, float  *s);



#define drotg32 FORTRAN_WRAPPER(drotg32)
EXTERN_C void drotg32(double *da, double *db, double *c, double *s);



#define crotg32 FORTRAN_WRAPPER(crotg32)
EXTERN_C void crotg32(creal32_T *ca, creal32_T *cb, creal32_T *c, creal32_T *s);



#define zrotg32 FORTRAN_WRAPPER(zrotg32)
EXTERN_C void zrotg32(creal_T *ca, creal_T *cb, creal_T *c, creal_T *s);



#define srot32 FORTRAN_WRAPPER(srot32)
EXTERN_C void srot32(const int *n32, float  *sx, const int *incx32, float  *sy, 
            const int *incy32, const float *c, const float *s);



#define drot32 FORTRAN_WRAPPER(drot32)
EXTERN_C void drot32(const int *n32, double *dx, const int *incx32, double *dy,
            const int *incy32, const double *c, const double *s);



#define csrot32 FORTRAN_WRAPPER(csrot32)
EXTERN_C void csrot32(const int *n32, creal32_T *cx, const int *incx32, creal32_T *cy, 
             const int *incy32, const float *c, const creal32_T *s);



#define zdrot32 FORTRAN_WRAPPER(zdrot32)
EXTERN_C void zdrot32(const int *n32, creal_T *cx, const int *incx32, creal_T *cy, 
             const int *incy32, const double *c, const creal_T *s);



#define sscal32 FORTRAN_WRAPPER(sscal32)
EXTERN_C void sscal32(const int *n32, const float *sa, float  *sx, const int *incx32);



#define dscal32 FORTRAN_WRAPPER(dscal32)
EXTERN_C void dscal32(const int *n32, const double *da, double *dx, const int *incx32);



#define cscal32 FORTRAN_WRAPPER(cscal32)
EXTERN_C void cscal32(const int *n32, const creal32_T *ca, creal32_T *cx, const int *incx32);



#define zscal32 FORTRAN_WRAPPER(zscal32)
EXTERN_C void zscal32(const int *n32, const creal_T *za, creal_T *zx, const int *incx32);



#define sswap32 FORTRAN_WRAPPER(sswap32)
EXTERN_C void sswap32(const int *n32, float  *sx, const int *incx32, float  *sy, 
             const int *incy32);



#define dswap32 FORTRAN_WRAPPER(dswap32)
EXTERN_C void dswap32(const int *n32, double *dx, const int *incx32, double *dy,
             const int *incy32);



#define cswap32 FORTRAN_WRAPPER(cswap32)
EXTERN_C void cswap32(const int *n32, creal32_T *cx, const int *incx32, 
             creal32_T *cy, const int *incy32);



#define zswap32 FORTRAN_WRAPPER(zswap32)
EXTERN_C void zswap32(const int *n32, creal_T *zx, const int *incx32, 
             creal_T *zy, const int *incy32);



#define strsm32 FORTRAN_WRAPPER(strsm32)
EXTERN_C void strsm32(char   *side, char   *uplo, char   *transa, 
             char *diag, const int *m32, const int *n32,
             const float *alpha, const float *a, const int *lda32, 
             float  *b, const int *ldb32);



#define dtrsm32 FORTRAN_WRAPPER(dtrsm32)
EXTERN_C void dtrsm32(char   *side, char   *uplo, char   *transa, 
             char *diag, const int *m32, const int *n32, 
             const double *alpha, const double *a, const int *lda32, 
             double *b, const int *ldb32);



#define ctrsm32 FORTRAN_WRAPPER(ctrsm32)
EXTERN_C void ctrsm32(char   *side, char   *uplo, char   *transa, 
             char *diag, const int *m32, const int *n32, 
             const creal32_T *alpha, const creal32_T *a, const int *lda32, 
             creal32_T *b, const int *ldb32);



#define ztrsm32 FORTRAN_WRAPPER(ztrsm32)
EXTERN_C void ztrsm32(char   *side, char   *uplo, char   *transa,
             char *diag, const int *m32, const int *n32,
             const creal_T *alpha, const creal_T *a, const int *lda32, 
             creal_T *b, const int *ldb32);



#define strsv32 FORTRAN_WRAPPER(strsv32)
EXTERN_C void strsv32(char   *uplo, char   *trans, char   *diag,
                const int *n32, const float *a, const int *lda32,
                float *x, const int *incx32);



#define dtrsv32 FORTRAN_WRAPPER(dtrsv32)
EXTERN_C void dtrsv32(char   *uplo, char   *trans, char   *diag,
                const int *n32, const double *a, const int *lda32, 
                double *x, const int *incx32);



#define ctrsv32 FORTRAN_WRAPPER(ctrsv32)
EXTERN_C void ctrsv32(char   *uplo, char   *trans, char   *diag, 
                const int *n32, const creal32_T *a, const int *lda32, 
                creal32_T *x, const int *incx32);



#define ztrsv32 FORTRAN_WRAPPER(ztrsv32)
EXTERN_C void ztrsv32(char   *uplo, char   *trans, char   *diag,
                const int *n32, const creal_T *a, const int *lda32, 
                creal_T *x, const int *incx32);


#endif /* blascompat32_h */
