#include "OSC/Mesh.hh"
#include "OSC/tmesh.h"

using namespace OSC;


int main(int argc, char **argv)
{
  //Mesh mesh;
  //mesh.readOFF(argv[1]);
  //mesh.computeLP(100);

  TMesh tmesh;
  tmesh.ReadOffFile(argv[1]);
  tmesh.computeLP(atoi(argv[2]));
}
