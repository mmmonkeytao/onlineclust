#include <math.h>
#include <cfloat>
#include <queue>
#include <fstream>

#include "OSC/tmesh.h"
#include "OSC/offobj.h"
#include "../arpack++/examples/areig/areig.h"


double __partcolorgrtable[31][3] = {
  {0, 0,       1},
  {0, 0.0625,  0.9375},
  {0, 0.1250,  0.8750},
  {0, 0.1875,  0.8125},
  {0, 0.2500,  0.7500},
  {0, 0.3125,  0.6875},
  {0, 0.3750,  0.6250},
  {0, 0.4375,  0.5625},
  {0, 0.5625,   0.4375},
  {0, 0.6250,   0.3750},
  {0, 0.6875,   0.3125},
  {0, 0.7500,   0.2500},
  {0, 0.8125,   0.1875},
  {0, 0.8750,   0.1250},
  {0, 0.9375,   0.0625},

  {0,       1,        0},
  {0.0625,  0.9375,   0},
  {0.1250,  0.8750,   0},
  {0.1875,  0.8125,   0},
  {0.2500,  0.7500,   0},
  {0.3125,  0.6875,   0},
  {0.3750,  0.6250,   0},
  {0.4375,  0.5625,   0},
  {0.5625,  0.4375,   0},
  {0.6250,  0.3750,   0},
  {0.6875,  0.3125,   0},
  {0.7500,  0.2500,   0},
  {0.8125,  0.1875,   0},
  {0.8750,  0.1250,   0},
  {0.9375,  0.0625,   0},
  {1,       0,        0}
};

void get_color(double v, double minv, double maxv, double c[3])
{
  double scale = 0.8;
  /*
  //---------------------------------------------------------------------------------------
  //all warm to cold
  double nnv = maxv - minv;
  int minc = 0, maxc = 63;
  double vs = (v - minv) / nnv * (maxc - minc + 1);

  int i = (int)(vs);
  for(int j = 0; j < 3; j ++)
  c[j] = scale * ((vs - i) * (__colortable[i + 1][j] - __colortable[i][j]) + __colortable[i][j]);

  //  cout<<"c: "<<c[0]<<" "<<c[1]<<" "<<c[2]<<std::endl;
  //---------------------------------------------------------------------------------------
  */

  //maxv = 2.5;
  //minv = 1;
  double nnv = maxv - minv;

  if(fabs(nnv) < MYNZERO){
    for(int j = 0; j < 3; j ++)
      c[j] = scale * __partcolorgrtable[15][j]; 		
    return;
  }
  
  //---------------------------------------------------------------------------------------
  //warm to cold
  int minc = 0, maxc = 30;
  if(v <= minv){
    for(int j = 0; j < 3; j ++)
      c[j] = scale * __partcolorgrtable[minc][j]; 
    return;
  }
  
  if(v >= maxv){
    for(int j = 0; j < 3; j ++)
      c[j] = scale * __partcolorgrtable[maxc][j];
    return; 
  }
  double vs = (v - minv) / nnv * (maxc - minc);

  int i = (int)(vs);
  for(int j = 0; j < 3; j ++)
    c[j] = scale * ((vs - i) * (__partcolorgrtable[i + 1][j] - __partcolorgrtable[i][j]) + __partcolorgrtable[i][j]);
  //  cout<<"c: "<<c[0]<<" "<<c[1]<<" "<<c[2]<<std::endl;
  //---------------------------------------------------------------------------------------

  /*
  //---------------------------------------------------------------------------------------
  //grey scale
  
  double maxc = 0, minc = 0.5;
  
  if(v <= minv) { c[0] = c[1] = c[2] = minc; return; }
  if(v >= maxv) { c[0] = c[1] = c[2] = maxc; return; }

  double t = (v - 1) / nnv;
  c[0] = c[1] = c[2] = minc  + (maxc - minc) * t;
  
  //---------------------------------------------------------------------------------------
  */  
}


bool TMesh::ReadOffFile(char *filename)
{
  std::string str(filename);
  OffFileReader reader(str);    
  
  if(reader.bad)
    return false;
    
  OffObj obj;
  if( reader.get_next_off_object(obj) ){
      
    for(unsigned int i = 0; i < obj.vertices.size(); i ++){
      add_vertex( VTMesh( obj.vertices[i](0), 
			  obj.vertices[i](1), 
			  obj.vertices[i](2)) );
    }
    for(unsigned int i = 0; i < obj.facets.size(); i ++){
    
      if(obj.facets[i].size() != 3){
	std::cerr << "Error: invalid triangle mesh." << std::endl;
        return false;
      }
      unsigned int fid = add_facet( FTMesh( (obj.facets[i])[0], 
					    (obj.facets[i])[1], 
					    (obj.facets[i])[2]) );
      assert(fid == i);
    }
    
    //get the bounding box of mesh
    GetBBox();
    //Generate the topology for tmesh
    GenerateMeshTopo();
    //Mark the non_manifoldness
    MarkNonManifoldness();
    //debug
    //PrintMeshTopo();
    //getchar();
    
    return true;
  }
  else {
    return false;
  }
}

bool TMesh::ReadOffFile(char *filename, bool wcolor)
{
  std::ifstream fin_temp;
  fin_temp.open(filename);
  if( !fin_temp.good() ){
    return false;
  }
  
  std::ofstream fout_temp;
  fout_temp.open("temp.m");
  if(!fout_temp.good()){
    std::cerr<<"Failed to open file temp.m" << std::endl;
    return false;
  }
  
  // Read the off file and skip the comments.
  // Write the comment-less off file to a file, called "temp.m".
  while(! fin_temp.eof()){
    char line[90];
    fin_temp.getline(line, 90);
    if(line[0] == 'O' || line[0] == '#')
      ;
    else
      fout_temp << line << std::endl;
  }
  fin_temp.close();
  fout_temp.close();
			  
	
  FILE *fp;
  if( (fp = fopen("temp.m", "r")) == NULL ){
    std::cerr << "Failed to open file temp.m" << std::endl;
    return false;
  }

  unsigned int n_ver, n_facet, n_edge;
  fscanf(fp, "%d %d %d", &n_ver, &n_facet, &n_edge);
  
  //  cerr<<"n_ver: "<<n_ver<<" n_facet: "<<n_facet<<endl;
  
  float x, y, z;
  //vertex information
  for(unsigned int i = 0; i < n_ver; i ++){
    fscanf(fp, "%f %f %f", &x, &y, &z);
    //cerr<<"x y z"<<x<<" "<<y<<" "<<z<<endl;
    //VTMesh v(x, y, z);
    add_vertex( VTMesh(x, y, z) );
  }		

  //cout<<"vertex done"<<endl;	
	
  //facet information	
  int nv, vid0, vid1, vid2;
  float r, g, b, a;
  for(unsigned int i = 0; i < n_facet; i ++){
    fscanf(fp, "%d %d %d %d", &nv, &vid0, &vid1, &vid2);
    if(wcolor) fscanf(fp, "%f %f %f %f", &r, &g, &b, &a);
	
    
    unsigned int fid = add_facet( FTMesh(vid0, vid1, vid2) );
    assert(fid == i);
  }
  //cerr<<"facet done"<<endl;

  assert(v_count() == n_ver);
  assert(f_count() == n_facet);
  
  //get the bounding box of mesh
  GetBBox();

  //Delete the temp.m file
  system("rm temp.m");

  //Generate the topology for tmesh
  GenerateMeshTopo();
  //Mark the non_manifoldness
  MarkNonManifoldness();
  //debug
  //  PrintMeshTopo();
  //  getchar();

  return true;
}

void TMesh::GenerateMeshTopo()
{
  unsigned int vid0, vid1, vid2;
  for(unsigned int i = 0; i < f_count(); i ++){
    vid0 = facet(i).vert(0);
    vid1 = facet(i).vert(1);
    vid2 = facet(i).vert(2);
    //    	cout<<i<<"th facet: vertices: "<<vid0<<" "<<vid1<<" "<<vid2<<endl;

    //generate the vertex topology
    vertex(vid0).add_facet(i);
    vertex(vid1).add_facet(i);
    vertex(vid2).add_facet(i);
    
    vertex(vid0).add_unique_vert(vid1);
    vertex(vid1).add_unique_vert(vid0);
    vertex(vid1).add_unique_vert(vid2);
    vertex(vid2).add_unique_vert(vid1);
    vertex(vid2).add_unique_vert(vid0);
    vertex(vid0).add_unique_vert(vid2);
  }
  //cerr<<"facet done"<<endl;

  for(unsigned int i = 0; i < f_count(); i ++)
    {
      //cout<<"fid: "<<i<<endl;
  
      //iterate over all facets	
      for(int j = 0; j < 3; j ++){
    	//cout<<j<<"th vertex "<<endl;
  
	//edge vert(j) -- vert((j + 1) % 3)
    	vid1 = facet(i).vert(j);
    	vid2 = facet(i).vert((j + 1) % 3);
	//seach the adjacent triangle sharing edge (j + 2) % 3
    	for(unsigned int fit = 0; fit < vertex(vid1).n_facets(); fit ++){
	  unsigned int fid = vertex(vid1).facet(fit);
	  if(fid <= i) continue;

	  //cout<<"connected facet: "<<fid<<endl;
			
	  int i1, i2;
	  if( (i2 = facet(fid).index(vid2)) == -1 ) continue;
	  i1 = facet(fid).index(vid1);
	  assert(i1 >= 0);			
			
	  if( facet(i).facet((j + 2) % 3) >= 0){
	    std::cerr << "non-manifold1: " << i <<"--" << fid
		      <<" along edge: "<<vid1<<" "<<vid2<<std::endl;
	    continue;
	  }  
			
	  for(int k = 0; k < 3; k ++){
	    if(k != i1 && k != i2){ 
	      if(facet(fid).facet(k) >= 0){
		std::cerr<<"non-manifold1: "<<i<<"--"
			 <<fid<<" along edge: "<<vid1<<" "<<vid2<<std::endl;
	      }  
	      else{ //Only when both facets have not neighbouring facet along 
		//this edge can they be the neighbouring faect for each other. 
		facet(fid).set_facet(k, i);
		facet(i).set_facet((j + 2) % 3, fid);
	      }  
	      break;
	    }
	  }//for k
	}//for fit
      }//for j
    }//for i
  //cout<<"topo done"<<endl;

  //set boundary flag
  //for particle
  for(unsigned int i = 0; i < f_count(); i ++){
    for(int j = 0; j < 3; j ++){
      if(facet(i).facet(j) >= 0) continue;

      facet(i).set_flag(FTMESH_FLAG_BOUNDARY);
      vertex( facet(i).vert((j + 1) % 3) ).set_flag(VTMESH_FLAG_BOUNDARY);
      vertex( facet(i).vert((j + 2) % 3) ).set_flag(VTMESH_FLAG_BOUNDARY);
    }
  }
  //Check whether the topology of the mesh is valid.
  CheckMeshTopo();
}

//--------------------------------------------------
//MarkNonManifoldness: 
//----------------
//Check the non_manifoldness for each vertex and mark 
//the vertex and all the incident facets if it is a 
//non_manifold vertex. 
//--------------------------------------------------
void TMesh::MarkNonManifoldness()
{
  unsigned int vid, fid, count;
  int ind, fid_circ, ind_circ, fid_pre;
  for(vid = 0; vid < v_count(); vid ++){
    VTMesh vert = vertex(vid);
    if( vert.n_facets() == 0) continue;
    fid = vert.facet(0);
    ind = facet(fid).index(vid);
    assert(ind >= 0);
    
    facet(fid).set_flag(FTMESH_FLAG_VISITED);
    count = 1;
    
#define UMBRELLAWALK(vid, fid, fid_circ)				\
    fid_pre = fid;							\
    while(fid_circ >= 0 && fid_circ != (int)fid){			\
      if( facet(fid_circ).check_flag(FTMESH_FLAG_VISITED) ) break;	\
      facet(fid_circ).set_flag(FTMESH_FLAG_VISITED);			\
      count ++;								\
      ind_circ = facet(fid_circ).index(vid);				\
      assert(ind_circ >= 0);						\
      if( fid_pre != facet(fid_circ).facet( (ind_circ + 1) % 3 ) ){	\
	fid_pre = fid_circ;						\
	fid_circ = facet(fid_circ).facet( (ind_circ + 1) % 3 );		\
      }									\
      else{								\
	fid_pre = fid_circ;						\
	fid_circ = facet(fid_circ).facet( (ind_circ + 2) % 3 );		\
      }									\
    }
   
    fid_circ = facet(fid).facet( (ind + 1) % 3 );
    UMBRELLAWALK(vid, fid, fid_circ);    

    fid_circ = facet(fid).facet( (ind + 2) % 3 );
    UMBRELLAWALK(vid, fid, fid_circ);
  
    //If the incident facets does not form an umbrella, then mark 
    if( count < vert.n_facets() ){
      vert.set_flag(VTMESH_FLAG_NMANIFOLD);
      for(unsigned int i = 0; i < vert.n_facets(); i ++)
        facet( vert.facet(i) ).un_set_flag(FTMESH_FLAG_NMANIFOLD);
    }
    
    //Unset FTMESH_FLAG_VISITED
    for(unsigned int i = 0; i < vert.n_facets(); i ++)
      facet( vert.facet(i) ).un_set_flag(FTMESH_FLAG_VISITED);
  }
}

//--------------------------------------------------
//OrientateFacets: 
//----------------
//Orientate the facets so that all the manifold facets will have
//a consistent orientation
//--------------------------------------------------
void TMesh::OrientateFacets()
{
  unsigned int fid, f;
  std::queue<unsigned int> fid_queue;

  unsigned int max_nfacets, nfacets, fid_start;
  unsigned int vid1, vid2;
  int ind1, ind2;
  int fid_adj;

  //Find largest part of the surface which can be orientated. 
  max_nfacets = 0;
  for(f = 0; f < f_count(); f ++){
    if( facet(f).check_flag(FTMESH_FLAG_NMANIFOLD) ) continue;
    if( facet(f).check_flag(FTMESH_FLAG_VISITED) ) continue;
    
    fid_queue.push(f);
    facet(f).set_flag(FTMESH_FLAG_VISITED);

    nfacets = 0;
    while( !fid_queue.empty() ){
      fid = fid_queue.front(); fid_queue.pop();
      nfacets ++;
      for(int j = 0; j < 3; j ++){
        fid_adj = facet(fid).facet(j);
        if(fid_adj < 0) continue;
        if( facet(fid_adj).check_flag(FTMESH_FLAG_NMANIFOLD) ) continue;
        if( facet(fid_adj).check_flag(FTMESH_FLAG_VISITED) ) continue;
        fid_queue.push(fid_adj);
        facet(fid_adj).set_flag(FTMESH_FLAG_VISITED);
      }
    }
    
    if( nfacets > max_nfacets ){
      max_nfacets = nfacets;
      fid_start = f;
    }
  }
  //unset flags
  for(f = 0; f < f_count(); f ++)
    facet(f).un_set_flag(FTMESH_FLAG_VISITED);
  
  
  //orientate the facets
  fid_queue.push(fid_start);
  facet(fid_start).set_flag(FTMESH_FLAG_ORIENTATE);
  while( !fid_queue.empty()){
    fid = fid_queue.front(); fid_queue.pop();
    for(int j = 0; j < 3; j ++){
      fid_adj = facet(fid).facet(j);
      if(fid_adj < 0) continue;
      if( facet(fid_adj).check_flag(FTMESH_FLAG_NMANIFOLD) ) continue;
      if( facet(fid_adj).check_flag(FTMESH_FLAG_ORIENTATE) ) continue;
      
      vid1 = facet(fid).vert( (j + 1) % 3 );
      vid2 = facet(fid).vert( (j + 2) % 3 );
      ind1 = facet(fid_adj).index(vid1);
      ind2 = facet(fid_adj).index(vid2);
      
      assert( ind1 >= 0 && ind2 >= 0 );
      
      //If the orientation of "fid" and "fid_adj" are consisitent
      if( (ind2 + 1) % 3 == ind1 ){
        if( facet(fid).check_flag(FTMESH_FLAG_REVERSE) )
          facet(fid_adj).set_flag(FTMESH_FLAG_REVERSE);
      }
      else{ //Otherwise the orientation of "fid" and "fid_adj" are NOT consisitent
        assert( (ind1 + 1) % 3  == ind2 );
        if( !(facet(fid).check_flag(FTMESH_FLAG_REVERSE)) )
          facet(fid_adj).set_flag(FTMESH_FLAG_REVERSE);
      } 
      fid_queue.push(fid_adj);
      facet(fid_adj).set_flag(FTMESH_FLAG_ORIENTATE);
    }//for j
  }//while
}

void TMesh::PrintMeshTopo()
{
  std::cout<<"vertex top"<<std::endl;
  for(unsigned int i = 0; i < v_count(); i ++){
    std::cout<<"vertex "<<i<<": "<<vertex(i).coord()<<std::endl;
    std::cout<<"\t incident vertex: ";
    for(unsigned int vit = 0; vit < vertex(i).n_verts(); vit ++)
      std::cout<<"\t"<<vertex(i).vert(vit)<<" ";
    std::cout<<std::endl;
    std::cout<<"\t incident triangle: ";
    for(unsigned int fit = 0; fit < vertex(i).n_facets(); fit ++)
      std::cout<<"\t"<<vertex(i).facet(fit)<<" ";
    std::cout<<std::endl;
      
  }
  std::cout<<std::endl;
  
  std::cout<<"triangle top"<<std::endl;
  for(unsigned int i = 0; i < f_count(); i ++){
    std::cout<<"triangle "<<i<<std::endl;
    
    std::cout<<"\t vert: ";
    for(int j = 0; j < 3; j ++)
      std::cout<<"\t"<<facet(i).vert(j)<<" ";
    std::cout<<std::endl;
    
    std::cout<<"\t adj_facet: ";
    for(int j = 0; j < 3; j ++)
      std::cout<<"\t"<<facet(i).facet(j)<<" ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
}

bool TMesh::CheckMeshTopo()
{
  //check facet topo
  bool good = true;
  for(unsigned int i = 0; i < f_count(); i ++){
    for(int j = 0; j < 3; j ++){
      int fid = facet(i).facet(j);
      if(fid < 0) continue;
      int findex = facet(fid).findex(facet(i).vert((j + 1) % 3), facet(i).vert((j + 2) % 3));
      if(findex < 0){
	std::cerr<<"mesh topo check error!"<<std::endl;
	std::cerr<<"fid: "<<i<<" eid: "<<" fid_adj: "<<fid<<std::endl;
        
        good = false;
      }
      //assert(findex >= 0);
    }
  }
	
  //check vertex topo	
  for(unsigned int i = 0; i < v_count(); i ++){
    for(unsigned int j = 0; j < vertex(i).n_facets(); j ++){
      unsigned int fid = vertex(i).facet(j);
      int vindex = facet(fid).index(i);

      if(vindex < 0){
	std::cerr<<"mesh topo check error!"<<std::endl;
	std::cerr<< "vid: "<<i<<" fid: "<<fid<<std::endl;

        good = false;
      }
      
      //assert(vindex >= 0);
    }

    for(unsigned int j = 0; j < vertex(i).n_verts(); j ++){
      unsigned int vid = vertex(i).vert(j);
      int vindex = vertex(vid).index(i);
      if(vindex < 0){
	std::cerr<<"mesh topo check error!"<<std::endl;
	std::cerr<<"vid: "<<i<<" vid: "<<vid<<std::endl;

        good = false;
      }
      //assert(vindex >= 0);
    }
  }

  if(!good){
    std::cerr<<"Warning: the input mesh is not a manifold, which MAY crash the successive computation."<<std::endl;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////
//GetBBox
//--------
//Get a box which is twice as bigger as the boundding box of the  object
//
void TMesh::GetBBox()
{
  if(v_count() == 0 ){
    _pmin = Eigen::Vector3d();
    _pmax = Eigen::Vector3d();
    return;
  }
  
  for(int i = 0; i < 3; i ++){
    _pmin[i] = (vertex(0).coord())(i);
    _pmax[i] = (vertex(0).coord())(i);
  }
  
  for(unsigned int i = 1; i < v_count(); i ++){
    for(int j = 0; j < 3; j ++){
      if( (vertex(i).coord())(j) < _pmin[j] )
        _pmin[j] = (vertex(i).coord())(j);
      else if((vertex(i).coord())(j) > _pmax[j])
        _pmax[j] = (vertex(i).coord())(j);
    }//for j
  }//for i

    
  //cerr<<"BBox: min "<<_pmin[0]<<" "<<_pmin[1]<<" "<<_pmin[2]
  //<<" max "<<_pmax[0]<<" "<<_pmax[1]<<" "<<_pmax[2]<<std::endl;
}

bool TMesh::IsOutBBox(Eigen::Vector3d const &pt)
{
  if(pt[0] < _pmin[0] || pt[0] > _pmax[0]) return true;
  if(pt[1] < _pmin[1] || pt[1] > _pmax[1]) return true;
  if(pt[2] < _pmin[2] || pt[2] > _pmax[2]) return true;
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////
//Meshsize
//--------
//Estimate mesh size
//
void TMesh::MeshSize(double& maxs, double& mins, double& aves)
{
  maxs = DBL_MIN;
  mins = DBL_MAX;
  aves = 0;
  for(unsigned int i = 0; i < f_count(); i ++){
    for(unsigned int j = 0; j < 3; j ++){
      int vid0 = facet(i).vert(j);
      int vid1 = facet(i).vert((j + 1) % 3);
      double length = (vertex(vid0).coord() - vertex(vid1).coord()).
	dot( vertex(vid0).coord() - vertex(vid1).coord());
									
      length = sqrt(fabs(length));
      aves += length;
      if(maxs < length){
	maxs = length;
      }
      if(mins > length){
	mins = length;
      }
    }			
  }
  aves /= 3 * f_count();
	
  std::cout<<"maxs: "<<maxs<<" mins: "<<mins<<" aves: "<<aves<<std::endl;
}


//================================================================
//begin: off file outpur function
//================================================================
void TMesh::OutMeshOffFile(char *filename)
{
  FILE *fp;
  if( (fp = fopen(filename, "w")) == NULL ){
    std::cout<<"Failed to open file!"<<std::endl;
    return;
  }
  fprintf(fp, "LIST\n");
  fprintf(fp, "appearance {linewidth 7}\n");
  fprintf(fp, "{\n");
  fprintf(fp, "OFF\n");
	
  fprintf(fp, "%d %d 0\n", v_count(), f_count());
  for(unsigned int i = 0; i < v_count(); i ++){
    fprintf(fp, "%f %f %f\n", (vertex(i).coord())(0), (vertex(i).coord())(1), (vertex(i).coord())(2));
  }
  for(unsigned int i = 0; i < f_count(); i ++){

    if(  vertex(facet(i).vert(0)).check_flag(VTMESH_FLAG_SELECTED) 
	 && vertex(facet(i).vert(1)).check_flag(VTMESH_FLAG_SELECTED)
	 && vertex(facet(i).vert(2)).check_flag(VTMESH_FLAG_SELECTED))
      fprintf(fp, "3 %d %d %d 0.2 0.2 0.2 1\n", facet(i).vert(0), facet(i).vert(1), facet(i).vert(2));
    else
      fprintf(fp, "3 %d %d %d 1 1 1 1\n", facet(i).vert(0), facet(i).vert(1), facet(i).vert(2));
  }
  fprintf(fp, "}\n");
	
  fclose(fp);
}

class ARMat
{
public:

  ARMat(TMesh::SparseMatrix const &Amat) : 
    _nnz(Amat.nonZeros()),
    _irow(new int [_nnz]),
    _pcol(new int [Amat.outerSize()+1]),
    _A(new double [_nnz])
  {
    uint i=0;
    for (int k=0; k<Amat.outerSize(); ++k){
      _pcol[k] = Amat.outerIndexPtr()[k];
      for (TMesh::SparseMatrix::InnerIterator it(Amat,k); it; ++it, ++i){
	_irow[i] = it.row();
	_A[i]    = it.value();
      }
    }
    _pcol[Amat.outerSize()] = _nnz;
  }

  ~ARMat()
  {
    delete [] _irow;
    delete [] _pcol;
    delete [] _A;
  }

  int computeGenEig(uint n, ARMat const &other, uint K, Eigen::VectorXd &evalR,
		   Eigen::VectorXd &evalI, Eigen::VectorXd &evec) const
  {
    evalR = Eigen::VectorXd(K);
    evalI = Eigen::VectorXd(K);
    evec  = Eigen::VectorXd(n*K*2);

    return AREig(evalR.data(), evalI.data(), evec.data(),
		 n, _nnz, _A, _irow, _pcol, 
		 other._nnz, other._A, other._irow, other._pcol, 1e-5, K);

  }

private:

  uint _nnz;
  int *_irow;
  int *_pcol;
  double *_A;  
};


void computeSpectrum(uint n, TMesh::SparseMatrix const &Amat, 
		     TMesh::SparseMatrix const &Bmat, uint K)
{
  ARMat A(Amat), B(Bmat);
  Eigen::VectorXd evalR(K), evalI(K), evec(2*K*n);

  int nconv = A.computeGenEig(n, B, K, evalR, evalI, evec);

  for(int i=0; i<(nconv < K ? nconv : K); ++i){
    std::cout << " lambda [" << i << "]: " << evalR(i);
    if(evalI(i) >= 0.0) {
      std::cout << " + " << evalI(i) << " I" << std::endl;
    }
    else
      std::cout << " - " << fabs(evalI(i)) << " I" << std::endl;
  }

  std::cout << "done " << std::endl;
}

void TMesh::computeLP(uint K)
{
  Eigen::VectorXd d;
  SparseMatrix W = -computeXuMeyerLaplace(d);
  uint n = d.size();

  std::vector<Eigen::Triplet<double> > tripletList;
  for(uint i=0; i<n; ++i)
    tripletList.push_back(Eigen::Triplet<double>(i,i,d(i)));
	
  SparseMatrix A(n, n);
  A.setFromTriplets(tripletList.begin(), tripletList.end());

  uint nb_evals = d.size() > K ? K : d.size();
  
  std::cout << "computing " << nb_evals << " eigenvalues.." << std::endl;
  computeSpectrum(n, W, A, nb_evals);
}



TMesh::SparseMatrix TMesh::computeXuMeyerLaplace(Eigen::VectorXd &ddv) const
{
  uint nv = v_count();
  uint nf = f_count();

 //compute the center and area for each facets
  std::vector<double> fareas(3*nf), fcots(3*nf);

  uint vid0, vid1, vid2;

  for(uint f = 0; f < nf; f ++){
    vid0 = facet(f).vert(0);
    vid1 = facet(f).vert(1);
    vid2 = facet(f).vert(2);

    Eigen::Vector3d vert0 = vertex(vid0).coord();
    Eigen::Vector3d vert1 = vertex(vid1).coord();    
    Eigen::Vector3d vert2 = vertex(vid2).coord();
    
    Eigen::Vector3d vv = (vert1 - vert0).cross(vert2 - vert0);
				
    double area = vv.norm();
    double l0 = sqrt( fabs((vert1 - vert2).dot(vert1 - vert2)) );
    double l1 = sqrt( fabs((vert0 - vert2).dot(vert0 - vert2)) );
    double l2 = sqrt( fabs((vert1 - vert0).dot(vert1 - vert0)) );

    double h0 = area / l0;
    double h1 = area / l1;
    double h2 = area / l2;

    fcots[3 * f] = sqrt(fabs(l1 * l1 - h2 * h2)) / h2;
    if( (l1 * l1 + l2 * l2 - l0 * l0) < 0){
      fcots[3 * f] = -fcots[3 * f];
    }
    fcots[3 * f + 1] = sqrt(fabs(l2 * l2 - h0 * h0)) / h0;
    if( (l0 * l0 + l2 * l2 - l1 * l1) < 0){
      fcots[3 * f + 1] = -fcots[3 * f + 1];
    }
    fcots[3 * f + 2] = sqrt(fabs(l0 * l0 - h1 * h1)) / h1;
    if( (l0 * l0 + l1 * l1 - l2 * l2) < 0 ){
      fcots[3 * f + 2] = -fcots[3 * f + 2];
    }

    if( fcots[3 * f] >= 0 && fcots[3 * f + 1] >= 0 && fcots[3 * f + 2] >= 0 ){
      fareas[3 * f] = 1.0 / 8.0 * (l1 * l1 * fcots[3 * f + 1] + l2 * l2 * fcots[3 * f + 2]);
      fareas[3 * f + 1] = 1.0 / 8.0 * (l0 * l0 * fcots[3 * f] + l2 * l2 * fcots[3 * f + 2]);
      fareas[3 * f + 2] = 1.0 / 8.0 * (l1 * l1 * fcots[3 * f + 1] + l0 * l0 * fcots[3 * f]);
    }
    else if(fcots[3 * f] < 0){
      fareas[3 * f] = area / 4;
      fareas[3 * f + 1] = area / 8;
      fareas[3 * f + 2] = area / 8;
    }
    else if(fcots[3 * f + 1] < 0){
      fareas[3 * f] = area / 8;
      fareas[3 * f + 1] = area / 4;
      fareas[3 * f + 2] = area / 8;
    }
    else{
      fareas[3 * f] = area / 8;
      fareas[3 * f + 1] = area / 8;
      fareas[3 * f + 2] = area / 4;
    }
  }

  std::vector<Eigen::Triplet<double> > tripletList;
  double Amix, totalweight;
  std::map<uint, double> adj_weight;

  ddv = Eigen::VectorXd(nv);

  for(uint i = 0; i < nv; ++i){
    Amix = 0;
    totalweight = 0;
    adj_weight.clear();

    for(uint kk = 0; kk < vertex(i).n_verts(); ++kk){
      adj_weight[vertex(i).vert(kk)] = 0;
    }

    for(uint kk = 0; kk < vertex(i).n_facets(); ++kk){
      uint f = vertex(i).facet(kk);
      int ind = facet(f).index(i);
      
      Amix += fareas[3 * f + ind];
      
      vid1 = facet(f).vert((ind + 1) % 3);
      vid2 = facet(f).vert((ind + 2) % 3);

      adj_weight.at(vid1) += fcots[3 * f + (ind + 2) % 3];
      adj_weight.at(vid2) += fcots[3 * f + (ind + 1) % 3];

      totalweight -= (fcots[3 * f + (ind + 2) % 3] + fcots[3 * f + (ind + 1) % 3]);
    }

    Amix *= 2;
    
    for(std::map<uint, double>::const_iterator iter = adj_weight.begin(); 
	iter != adj_weight.end(); ++iter){

      tripletList.push_back(Eigen::Triplet<double>(i, iter->first, iter->second));
    }

    tripletList.push_back(Eigen::Triplet<double>(i, i, totalweight));    
    ddv(i) = Amix;
  }	

  SparseMatrix mat(nv, nv);
  mat.setFromTriplets(tripletList.begin(), tripletList.end());

  return mat;
}


void TMesh::OutMeshOffFile(FILE *fp, double r, double g, double b, double a)
{
  if( fp == NULL ){
    std::cout<<"Invalid FILE pointer"<<std::endl;
    return;
  }
  
  fprintf(fp, "{\n");
  fprintf(fp, "OFF\n");
  fprintf(fp, "%d %d 0\n", v_count(), f_count());
  for(unsigned int i = 0; i < v_count(); i ++){
    fprintf(fp, "%f %f %f\n", (vertex(i).coord())(0), (vertex(i).coord())(1), (vertex(i).coord())(2));
  }
  for(unsigned int i = 0; i < f_count(); i ++){
    if( facet(i).check_flag(FTMESH_FLAG_SELECT) )
      fprintf(fp, "3 %d %d %d 1 0 0 1\n", facet(i).vert(0), facet(i).vert(1), facet(i).vert(2));
    else 
      fprintf(fp, "3 %d %d %d %f %f %f %f\n", facet(i).vert(0), facet(i).vert(1), facet(i).vert(2), r, g, b, a);
  }
  fprintf(fp, "}\n");
}
/*
  void TMesh::OutMeshOffFile(FILE *fp, double r, double g, double b, double a)
  {
  if( fp == NULL ){
  std::cout<<"Invalid FILE pointer"<<std::endl;
  return;
  }
  
  fprintf(fp, "{\n");
  fprintf(fp, "OFF\n");
  fprintf(fp, "%d %d 0\n", v_count(), f_count());
  for(unsigned int i = 0; i < v_count(); i ++){
  fprintf(fp, "%f %f %f\n", (vertex(i).coord())(0), (vertex(i).coord())(1), (vertex(i).coord())(2));
  }
  for(unsigned int i = 0; i < f_count(); i ++){

  fprintf(fp, "3 %d %d %d 1 0 0 1\n", facet(i).vert(0), facet(i).vert(1), facet(i).vert(2));
  fprintf(fp, "3 %d %d %d %f %f %f %f\n", facet(i).vert(0), facet(i).vert(1), facet(i).vert(2), r, g, b, a);
  }
  fprintf(fp, "}\n");
  }
*/
//================================================================
//end: off file outpur function
//================================================================


