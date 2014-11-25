#include <iostream>
#include "ompcore.h"

int main(){

   double *D, *x, *DtX, *XtX, *G;
   mwSize n=0,m=0,L=0;
   int T = 2;
   double eps = 10e-3;
   int gamma_mode = 1;
   int profile = 1;
	double msg_delta = 2;
	int erroromp = 2;

 mxArray* A = ompcore(D, x,DtX,XtX,G, n, m,L,T, eps, gamma_mode,  profile, msg_delta, erroromp);


	return 0;
}
