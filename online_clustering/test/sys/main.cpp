#include "hmp.h"
#include <iostream>

using namespace std;
using namespace onlineclust;

int main(){
  
  HMP test;

  MatrixXd c;
  test.hmp_core(c);


  return 0;
}
