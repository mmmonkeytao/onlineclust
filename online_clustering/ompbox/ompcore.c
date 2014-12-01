/**************************************************************************
 *
 * File name: ompcore.c
 *
 * Ron Rubinstein
 * Computer Science Department
 * Technion, Haifa 32000 Israel
 * ronrubin@cs
 *
 * Last Updated: 25.8.2009
 *
 *************************************************************************/

#include "ompcore.h"
#include "omputils.h"
#include "ompprof.h"
#include "myblas.h"
#include <math.h>
#include <string.h>


/******************************************************************************
 *                                                                            *
 *                           Batch-OMP Implementation                         *
 *                                                                            *
 ******************************************************************************/  
/**************************************************************************
 * Perform Batch-OMP or OMP-Cholesky on a specified set of signals, using
 * either a fixed number of atoms or an error bound.
 *
 * Parameters (not all required):
 *
 *   D - the dictionary, of size n X m
 *   x - the signals, of size n X L
 *   DtX - D'*x, of size m X L
 *   XtX - squared norms of the signals in x, sum(x.*x), of length L
 *   G - D'*D, of size m X m
 *   T - target sparsity, or maximal number of atoms for error-based OMP
 *   eps - target residual norm for error-based OMP
 *   gamma_mode - one of the constants FULL_GAMMA or SPARSE_GAMMA
 *   profile - if non-zero, profiling info is printed
 *   msg_delta - positive: the # of seconds between status prints, otherwise: nothing is printed
 *   erroromp - if nonzero indicates error-based OMP, otherwise fixed sparsity OMP
 *
 * Usage:
 *
 *   The function can be called using different parameters, and will have
 *   different complexity depending on the parameters specified. Arrays which
 *   are not specified should be passed as null (0). When G is specified, 
 *   Batch-OMP is performed. Otherwise, OMP-Cholesky is performed.
 *
 *   Fixed-sparsity usage:
 *   ---------------------
 *   Either DtX, or D and x, must be specified. Specifying DtX is more efficient.
 *   XtX does not need to be specified.
 *   When D and x are specified, G is not required. However, not providing G
 *   will significantly degrade efficiency.
 *   The number of atoms must be specified in T. The value of eps is ignored.
 *   Finally, set erroromp to 0.
 *
 *   Error-OMP usage:
 *   ----------------
 *   Either DtX and Xtx, or D and x, must be specified. Specifying DtX and XtX
 *   is more efficient.
 *   When D and x are specified, G is not required. However, not providing G
 *   will significantly degrade efficiency.
 *   The target error must be specified in eps. A hard limit on the number
 *   of atoms can also be specified via the parameter T. Otherwise, T should 
 *   be negative. Finally, set erroromp to nonzero.
 *
 *
 * Returns: 
 *   An mxArray containing the sparse representations of the signals in x
 *   (allocated using the appropriate mxCreateXXX() function).
 *   The array is either full or sparse, depending on gamma_mode.
 *
 **************************************************************************/

mxArray* ompcore(double D[], double x[], double DtX[], double XtX[], double G[], mwSize n, mwSize m, mwSize L,
                 int T, double eps, int gamma_mode, int profile, double msg_delta, int erroromp)
{
  
  profdata pd;
  mxArray *Gamma;
  mwIndex i, j, signum, pos, *ind, *gammaIr, *gammaJc, gamma_count;
  mwSize allocated_coefs, allocated_cols;
  int DtX_specified, XtX_specified, batchomp, standardomp, *selected_atoms;
  double *alpha, *r, *Lchol, *c, *Gsub, *Dsub, sum, *gammaPr, *tempvec1, *tempvec2; 
  double eps2, resnorm, delta, deltaprev, secs_remain;
  int mins_remain, hrs_remain;
  clock_t lastprint_time, starttime;
 
  
  
  /*** status flags ***/
  
  DtX_specified = (DtX!=0);   /* indicates whether D'*x was provided */
  XtX_specified = (XtX!=0);   /* indicates whether sum(x.*x) was provided */
  
  standardomp = (G==0);       /* batch-omp or standard omp are selected depending on availability of G */
  batchomp = !standardomp;
  
  
  
  /*** allocate output matrix ***/
  
  
  if (gamma_mode == FULL_GAMMA) {
    
    /* allocate full matrix of size m X L */
    
    Gamma = mxCreateDoubleMatrix(m, L, mxREAL);
    gammaPr = mxGetPr(Gamma);
    gammaIr = 0;
    gammaJc = 0;
  }
  else {
    
    /* allocate sparse matrix with room for allocated_coefs nonzeros */
    
    /* for error-omp, begin with L*sqrt(n)/2 allocated nonzeros, otherwise allocate L*T nonzeros */
    allocated_coefs = erroromp ? (mwSize)(ceil(L*sqrt((double)n)/2.0) + 1.01) : L*T;
    Gamma = mxCreateSparse(m, L, allocated_coefs, mxREAL);
    gammaPr = mxGetPr(Gamma);
    gammaIr = mxGetIr(Gamma);
    gammaJc = mxGetJc(Gamma);
    gamma_count = 0;
    gammaJc[0] = 0;
  }
  
  
  /*** helper arrays ***/
  
  alpha = (double*)mxMalloc(m*sizeof(double));        /* contains D'*residual */
  ind = (mwIndex*)mxMalloc(n*sizeof(mwIndex));        /* indices of selected atoms */
  selected_atoms = (int*)mxMalloc(m*sizeof(int));     /* binary array with 1's for selected atoms */
  c = (double*)mxMalloc(n*sizeof(double));            /* orthogonal projection result */
  
  /* current number of columns in Dsub / Gsub / Lchol */
  allocated_cols = erroromp ? (mwSize)(ceil(sqrt((double)n)/2.0) + 1.01) : T;
  
  /* Cholesky decomposition of D_I'*D_I */
  Lchol = (double*)mxMalloc(n*allocated_cols*sizeof(double));

  /* temporary vectors for various computations */
  tempvec1 = (double*)mxMalloc(m*sizeof(double));
  tempvec2 = (double*)mxMalloc(m*sizeof(double));
  
  if (batchomp) {
    /* matrix containing G(:,ind) - the columns of G corresponding to the selected atoms, in order of selection */
    Gsub = (double*)mxMalloc(m*allocated_cols*sizeof(double));
  }
  else {
    /* matrix containing D(:,ind) - the selected atoms from D, in order of selection */
    Dsub = (double*)mxMalloc(n*allocated_cols*sizeof(double));
    
    /* stores the residual */
    r = (double*)mxMalloc(n*sizeof(double));        
  }
  
  if (!DtX_specified) {
    /* contains D'*x for the current signal */
    DtX = (double*)mxMalloc(m*sizeof(double));  
  }
  
  
  
  /*** initializations for error omp ***/
  
  if (erroromp) {
    eps2 = eps*eps;        /* compute eps^2 */
    if (T<0 || T>n) {      /* unspecified max atom num - set max atoms to n */
      T = n;
    }
  }
  
  
  
  /*** initialize timers ***/
  
  initprofdata(&pd);             /* initialize profiling counters */
  starttime = clock();           /* record starting time for eta computations */
  lastprint_time = starttime;    /* time of last status display */
  
  
  
  /**********************   perform omp for each signal   **********************/
  
  
  
  for (signum=0; signum<L; ++signum) {
    
    
    /* initialize residual norm and deltaprev for error-omp */
    
    if (erroromp) {
      if (XtX_specified) {
        resnorm = XtX[signum];
      }
      else {
        resnorm = dotprod(x+n*signum, x+n*signum, n);
        addproftime(&pd, XtX_TIME);
      }
      deltaprev = 0;     /* delta tracks the value of gamma'*G*gamma */
    }
    else {
      /* ignore residual norm stopping criterion */
      eps2 = 0;
      resnorm = 1;
    }
    
    
    if (resnorm>eps2 && T>0) {
      
      /* compute DtX */
      
      if (!DtX_specified) {
        matT_vec(1, D, x+n*signum, DtX, n, m);
        addproftime(&pd, DtX_TIME);
      }
      
      
      /* initialize alpha := DtX */
      
      memcpy(alpha, DtX + m*signum*DtX_specified, m*sizeof(double));
      
      
      /* mark all atoms as unselected */
      
      for (i=0; i<m; ++i) {
        selected_atoms[i] = 0;
      }
      
    }
    

    /* main loop */
    
    i=0;
    while (resnorm>eps2 && i<T) {

      /* index of next atom */
      
      pos = maxabs(alpha, m);
      addproftime(&pd, MAXABS_TIME);
      
      
      /* stop criterion: selected same atom twice, or inner product too small */
      
      if (selected_atoms[pos] || alpha[pos]*alpha[pos]<1e-14) {
        break;
      }
      
      
      /* mark selected atom */
      
      ind[i] = pos;
      selected_atoms[pos] = 1;
      
      
      /* matrix reallocation */
      
      if (erroromp && i>=allocated_cols) {
        
        allocated_cols = (mwSize)(ceil(allocated_cols*MAT_INC_FACTOR) + 1.01);
        
        Lchol = (double*)mxRealloc(Lchol,n*allocated_cols*sizeof(double));
        
        batchomp ? (Gsub = (double*)mxRealloc(Gsub,m*allocated_cols*sizeof(double))) :
                   (Dsub = (double*)mxRealloc(Dsub,n*allocated_cols*sizeof(double))) ;
      }
      
      
      /* append column to Gsub or Dsub */
      
      if (batchomp) {
        memcpy(Gsub+i*m, G+pos*m, m*sizeof(double));
      }
      else {
        memcpy(Dsub+i*n, D+pos*n, n*sizeof(double));
      }
      
      
      /*** Cholesky update ***/
      
      if (i==0) {
        *Lchol = 1;
      }
      else {
        
        /* incremental Cholesky decomposition: compute next row of Lchol */
        
        if (standardomp) {
          matT_vec(1, Dsub, D+n*pos, tempvec1, n, i);      /* compute tempvec1 := Dsub'*d where d is new atom */
          addproftime(&pd, DtD_TIME);
        }
        else {
          vec_assign(tempvec1, Gsub+i*m, ind, i);          /* extract tempvec1 := Gsub(ind,i) */
        }
        backsubst('L', Lchol, tempvec1, tempvec2, n, i);   /* compute tempvec2 = Lchol \ tempvec1 */
        for (j=0; j<i; ++j) {                              /* write tempvec2 to end of Lchol */
          Lchol[j*n+i] = tempvec2[j];
        }
        
        /* compute Lchol(i,i) */
        sum = 0;
        for (j=0; j<i; ++j) {         /* compute sum of squares of last row without Lchol(i,i) */
          sum += SQR(Lchol[j*n+i]);
        }
        if ( (1-sum) <= 1e-14 ) {     /* Lchol(i,i) is zero => selected atoms are dependent */
          break;
        }
        Lchol[i*n+i] = sqrt(1-sum);
      }
      
      addproftime(&pd, LCHOL_TIME);

      i++;
      
      
      /* perform orthogonal projection and compute sparse coefficients */
      
      vec_assign(tempvec1, DtX + m*signum*DtX_specified, ind, i);   /* extract tempvec1 = DtX(ind) */
      cholsolve('L', Lchol, tempvec1, c, n, i);                     /* solve LL'c = tempvec1 for c */
      addproftime(&pd, COMPCOEF_TIME);
      

      /* update alpha = D'*residual */
      
      if (standardomp) {
        mat_vec(-1, Dsub, c, r, n, i);             /* compute r := -Dsub*c */
        vec_sum(1, x+n*signum, r, n);              /* compute r := x+r */
        
        
        /*memcpy(r, x+n*signum, n*sizeof(double));   /* assign r := x */
        /*mat_vec1(-1, Dsub, c, 1, r, n, i);         /* compute r := r-Dsub*c */
        
        addproftime(&pd, COMPRES_TIME);
        matT_vec(1, D, r, alpha, n, m);            /* compute alpha := D'*r */
        addproftime(&pd, DtR_TIME);
        
        /* update residual norm */
        if (erroromp) {
          resnorm = dotprod(r, r, n);
          addproftime(&pd, UPDATE_RESNORM_TIME);
        }
      }
      else {
        mat_vec(1, Gsub, c, tempvec1, m, i);                              /* compute tempvec1 := Gsub*c */
        memcpy(alpha, DtX + m*signum*DtX_specified, m*sizeof(double));    /* set alpha = D'*x */
        vec_sum(-1, tempvec1, alpha, m);                                  /* compute alpha := alpha - tempvec1 */
        addproftime(&pd, UPDATE_DtR_TIME);
        
        /* update residual norm */
        if (erroromp) {
          vec_assign(tempvec2, tempvec1, ind, i);      /* assign tempvec2 := tempvec1(ind) */
          delta = dotprod(c,tempvec2,i);               /* compute c'*tempvec2 */
          resnorm = resnorm - delta + deltaprev;       /* residual norm update */
          deltaprev = delta;
          addproftime(&pd, UPDATE_RESNORM_TIME);
        }
      }
    }
    
    
    /*** generate output vector gamma ***/

    if (gamma_mode == FULL_GAMMA) {    /* write the coefs in c to their correct positions in gamma */
      for (j=0; j<i; ++j) {
        gammaPr[m*signum + ind[j]] = c[j];
      }
    }
    else {
      /* sort the coefs by index before writing them to gamma */
      quicksort(ind,c,i);
      addproftime(&pd, INDEXSORT_TIME);
      
      /* gamma is full - reallocate */
      if (gamma_count+i >= allocated_coefs) {
        
        while(gamma_count+i >= allocated_coefs) {
          allocated_coefs = (mwSize)(ceil(GAMMA_INC_FACTOR*allocated_coefs) + 1.01);
        }
        
        mxSetNzmax(Gamma, allocated_coefs);
        mxSetPr(Gamma, (double*)mxRealloc(gammaPr, allocated_coefs*sizeof(double)));
        mxSetIr(Gamma, (mwIndex*)mxRealloc(gammaIr, allocated_coefs*sizeof(mwIndex)));
        
        gammaPr = mxGetPr(Gamma);
        gammaIr = mxGetIr(Gamma);
      }
      
      /* append coefs to gamma and update the indices */
      for (j=0; j<i; ++j) {
        gammaPr[gamma_count] = c[j];
        gammaIr[gamma_count] = ind[j];
        gamma_count++;
      }
      gammaJc[signum+1] = gammaJc[signum] + i;
    }
    
    
    
    /*** display status messages ***/
    
    if (msg_delta>0 && (clock()-lastprint_time)/(double)CLOCKS_PER_SEC >= msg_delta)
    {
      lastprint_time = clock();
      
      /* estimated remainig time */
      secs2hms( ((L-signum-1)/(double)(signum+1)) * ((lastprint_time-starttime)/(double)CLOCKS_PER_SEC) ,
        &hrs_remain, &mins_remain, &secs_remain);
      
      printf("omp: signal %d / %d, estimated remaining time: %02d:%02d:%05.2f\n",        
        signum+1, L, hrs_remain, mins_remain, secs_remain);
      mexEvalString("drawnow;");
    }
    
  }
  
  /* end omp */
  
  
  
  /*** print final messages ***/
  
  if (msg_delta>0) {
    printf("omp: signal %d / %d\n", signum, L);
  }
  
  if (profile) {
    printprofinfo(&pd, erroromp, batchomp, L);
  }
  
  
  
  /* free memory */
  
  if (!DtX_specified) {
    free(DtX);
  }
  if (standardomp) {
    free(r);
    free(Dsub);
  }
  else {
    free(Gsub);
  }  
  free(tempvec2);
  free(tempvec1);
  free(Lchol);
  free(c);
  free(selected_atoms);
  free(ind);
  free(alpha);
  
  return Gamma;
}
