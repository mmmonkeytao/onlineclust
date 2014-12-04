#include "hmp.h"
#include <iostream>

using namespace std;
using namespace onlineclust;

int main(){
  
  HMP hmpObj;

  MatrixXd x{50,51}, fea;
  x.setRandom();
  uint splevel[2] = {5, 10};
  hmpObj.hmp_core(x,"rgb",splevel, fea);
  
  return 0;
}
