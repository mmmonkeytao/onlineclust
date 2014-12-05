#include "OnlineStarClustering.h"
#include <stdexcept>


using namespace std;
using namespace onlineclust;

OnlineStarClustering::OnlineStarClustering(double sigma): 
  _data(), _similarityMatrix(0,0), _sigma(sigma), _spData(),_feaSize()
{}

OnlineStarClustering::OnlineStarClustering(double sigma, uint feaSize): 
  _data(), _similarityMatrix(0,0), _sigma(sigma), _spData(),_feaSize(feaSize)
{}

OnlineStarClustering::~OnlineStarClustering()
{
  // clear allocated data

}

void OnlineStarClustering::insert(VectorXd &vec)
{
  if(static_cast<uint>(vec.size()) != _feaSize)
    throw runtime_error("\nFeature inserted has wrong dimention.\n");
  
  // insert into sparse dataset
  _data.push_back(std::move(vec));
 
  // insert and update clusters
  insertSparseData();
}

void OnlineStarClustering::insertSparseData()
{
  if(_data.size() == 0) // no data to insert
    throw runtime_error("\nDataset is empty, nothing to insert.\n");

  list<uint> L;
  uint new_point_id = _data.size() - 1;
    
  for(auto &it: _elemInGraphSigma){
    //uint old_point_idx = _id2idx[old_vertex_id];

    double similarityValue = _data[it].dot( _data[new_point_id]);

    if (similarityValue >= _sigma)        
      L.push_back(it);
  }

  fastInsert(new_point_id, L);
  _elemInGraphSigma.push_back(new_point_id);
  //_id2idx[new_vertex_id] = new_point_idx;
  L.clear();

}

double OnlineStarClustering::computeSimilarity(SparseVectorXd const &x1,SparseVectorXd const &x2)const 
{
  // cosine similarrty
  
  //SparseVectorXd x{} = x1 - x2;
  return 0.8;//(x1-x2).norm();
} 

void OnlineStarClustering::loadAndAddData(char const *filename)
{
  ifstream ifile(filename);

  Eigen::VectorXd vec;
  ///////////////////////////
  double label;  int dataID = 0;
  ///////////////////////////

  // data dimension
  int dim  = 0, start_idx = _data.size();
  if(_data.size() != 0)
    dim = _data.back().size();

  do {
    vec = readVector(ifile);
    if(dim == 0)
      if(vec.size() == 0)
	throw "Could not read data file";
      else
	dim = vec.size();

    else if (dim != vec.size() && vec.size() != 0)
      throw "Invalid length of data vector";

    if(vec.size()){
      //vec /= vec.norm();
      ////////////////////////////////////////
      Eigen::VectorXd vector(vec.size()-1);
      label = vec(0);
      for(int i = 0; i < vec.size() - 1; i++){
        vector(i) = vec(i+1);
      }

      ////////////////////////////////////////
      _data.push_back(vector);
      //////////////////////////
      _labels[dataID] = label;
      dataID++;
      //////////////////////////
    }
  } while (vec.size() != 0);

  //updateSimilarityMatrix(start_idx);
  insertData(start_idx);
  //cout<<"after insert data: "<< _graph.size()<<endl;
  /*
  for(map<uint,Vertex>::const_iterator it1 = _graph.begin(); 
      it1 != _graph.end(); ++it1){

      list<uint> adj_list = it1->second.getAdjVerticesList();

      cout << "Point " << it1->first << " Adj list:"<<endl;
      for(list<uint>::const_iterator it2 = adj_list.begin(); 
	  it2 != adj_list.end(); ++it2)
	cout << *it2 << " ";

      cout << endl;
  }
  */

}

void OnlineStarClustering::addPoint(Eigen::VectorXd const &p, uint idx)
{
   // data dimension
  int dim  = 0, start_idx = _data.size();
  if(_data.size() == 0)
    dim = p.size();
  else
    dim = _data.back().size();
  
  if (dim != p.size())
    throw "Invalid length of data vector";

  _data.push_back(p);

  updateSimilarityMatrix(start_idx);
  insertLastPoint(idx);
}

void OnlineStarClustering::clear()
{
  _data.clear();
  _similarityMatrix.resize(0,0);

  _graph.clear();
  _id2idx.clear();
  _elemInGraphSigma.clear();
}


void OnlineStarClustering::exportDot(char const *filename, bool use_data) const
{
  ofstream ofile(filename);

  ofile << "digraph {" << endl;
  if(use_data)
    ofile << "overlap = true;" << endl; 
  else
    ofile << "overlap = false;" << endl; 
  ofile << "splines = true;" << endl; 
  ofile << "size = \"20,30\";" << endl;
  
  for(map<uint,Vertex>::const_iterator it = _graph.begin(); 
      it != _graph.end(); ++it){

    Vertex v = it->second;
    if(v.getType() == Vertex::CENTER)
      
      ofile << it->first << "  [shape = doublecircle,style=filled,"
	    << "fontsize = 20,label=\"" << it->first << "\"";
    
    else 
      ofile << it->first << "  [shape = circle,fontsize = 20,label=\""
	    << it->first << "\"";
    
    if(use_data){
      uint point_idx = _id2idx.at(it->first);
      ofile << ",pos = \""  << 10*_data[point_idx][0] << "," 
	    << 10*_data[point_idx][1] << "!\"";

    }
    ofile << "]" << endl;
  }

  ofile << "edge [dir=none]" << std::endl;
  for(map<uint,Vertex>::const_iterator it1 = _graph.begin(); 
      it1 != _graph.end(); ++it1){

    if(it1->second.getType() == Vertex::CENTER){
      list<uint> adj_list = it1->second.getAdjVerticesList();

      for(list<uint>::const_iterator it2 = adj_list.begin(); 
	  it2 != adj_list.end(); ++it2)
	ofile << it1->first << " -> " << *it2 << ";" << endl;
    }
  }

  ofile << "}" << endl;
  ofile.close();

}

double OnlineStarClustering::computeSimilarity(Eigen::VectorXd const &x1,
					       Eigen::VectorXd const &x2) const
{
  return x1.dot(x2) / (x1.norm() * x2.norm());
}

void OnlineStarClustering::updateSimilarityMatrix(uint start_idx)
{
  std::cout << "Computing similarity matrix..." << std::endl;
  //uint rows = _similarityMatrix.rows();

  _similarityMatrix.conservativeResize(_data.size(), _data.size());

  for(uint i=0; i<_data.size(); ++i){
    uint start_j = (start_idx > i) ? start_idx : i;
    for(uint j = start_j; j<_data.size(); ++j){
      _similarityMatrix(i,j) = _similarityMatrix(j,i) = computeSimilarity(_data[i],_data[j]);
    }
  }

  std::cout << "done." << std::endl;

  if(_similarityMatrix.rows() < 25)
    std::cout << _similarityMatrix << std::endl;
  else
    std::cout << _similarityMatrix.topLeftCorner<25,25>() << std::endl;
}


void OnlineStarClustering::insertData(uint start_idx)
{
  list<uint> L;
  
  for(uint dataID = start_idx; dataID < _data.size(); ++dataID){
    
    for(list<uint>::iterator it = _elemInGraphSigma.begin();
	it != _elemInGraphSigma.end(); ++it){
      
      uint vertex_id = *it;
      uint point_idx = _id2idx[vertex_id];
      double similarityValue = computeSimilarity(_data[point_idx], _data[dataID]);
      if ( similarityValue >= _sigma)        
	L.push_back(vertex_id);
    }

    fastInsert(dataID, L);
    _elemInGraphSigma.push_back(dataID);
    _id2idx[dataID] = dataID;
    L.clear();
  }
}

// insert last data point into graph
void OnlineStarClustering::insertLastPoint(uint new_vertex_id)
{
  if(_data.size() == 0) // no data to insert
    return;

  list<uint> L;
  uint new_point_idx = _data.size() - 1;

  for(list<uint>::iterator it = _elemInGraphSigma.begin();
      it != _elemInGraphSigma.end(); ++it){
      
      uint old_vertex_id = *it;
      uint old_point_idx = _id2idx[old_vertex_id];
      double similarityValue =  computeSimilarity(_data[old_point_idx], _data.back());
      if ( similarityValue >= _sigma)        
	L.push_back(old_vertex_id);
  }
  
  fastInsert(new_vertex_id, L);
  _elemInGraphSigma.push_back(new_vertex_id);
  _id2idx[new_vertex_id] = new_point_idx;
 }

Eigen::VectorXd OnlineStarClustering::readVector(std::ifstream &ifile) const
{
  string line;
  getline(ifile, line);
  
  size_t pos = 0;
  std::vector<double> vals;

  do {
    size_t next_pos    = line.find_last_of(' ',line.find_first_of(' ', pos+1) + 1);
    std::string substr = line.substr(pos, next_pos - pos);
    double val;

    if(substr == "inf" || substr == "INF")
      vals.push_back(HUGE_VAL);
    else {
      if(sscanf(substr.c_str(), "%lf", &val) == 1){
	       vals.push_back(val);
      }
    }
    
    pos = next_pos;
  } while(pos && pos < line.size());
  
  Eigen::VectorXd vec(vals.size());
  for(uint i=0; i<vals.size(); ++i)
    vec(i) = vals[i];
  
  return vec;
}


void OnlineStarClustering::fastInsert(uint alphaID, list <uint> &L)
{    
  // Make a vertex and insert in graph sigma.
  Vertex alpha;
  alpha.setID(alphaID);
    
  alpha.setDomCenterNull();
  alpha.setInQStatus(false);
    
  _graph.insert(pair<uint, Vertex>(alphaID ,alpha));
    
  uint betaID, betaDomCenterID, alphaDomCenterID;
    
  // For all beta in list L.
  for (auto &it: L){
    
    betaID = it;

    // Increment degrees
    _graph[alphaID].incrementDegree();
    _graph[betaID].incrementDegree();
    
    // Insert alphaID and betaID in each other's adjacency lists.
    _graph[alphaID].insertAdjVertex(betaID);
    _graph[betaID].insertAdjVertex(alphaID);
    
    ///////
    // list<uint> DomSL, AdjCL;
    // uint DomCenterID;

    // if (!_graph[betaID].isDomCenterNull()){
    //   betaDomCenterID = _graph[betaID].getDomCenter();
    //   DomSL = _graph[betaDomCenterID].getDomSatsList();

    //   for (list<uint>::iterator iter1 = DomSL.begin(); iter1 != DomSL.end(); ++iter1){
    //     DomCenterID = *iter1;
    //     AdjCL = _graph[DomCenterID].getAdjCentersList();
    //     sortList(AdjCL);
    //   }
    // }  
    ///////

    // Update center adjacency list if beta was a center.
    if (_graph[betaID].getType() == Vertex::CENTER) {
      _graph[alphaID].insertAdjCenter(betaID);
    }
    else {
      // **** CHANGE FOR FAST VERSION ***
      // Get degree of beta's dominant center
      betaDomCenterID = _graph[betaID].getDomCenter();
      
      // Insert if deg of beta has exceeded the one of its dom center.
      if ( _graph[betaID].getDegree() > _graph[betaDomCenterID].getDegree() ) {
	
	// Insert beta into the priority queue.
	_graph[betaID].setInQStatus(true);
	_priorityQ.push(_graph[betaID]);
      }    
    }
  } // List iteration ends.
  
  
  // **** CHANGE FOR FAST VERSION ***
    
  // If alpha's adjacent list is empty
  if (_graph[alphaID].isAdjCentersListEmpty()){
    
    // Insert alpha into the priority queue.
    _graph[alphaID].setInQStatus(true);
    _priorityQ.push(_graph[alphaID]); 
  }
    
  else {
    // Find alpha's dominant center.
    alphaDomCenterID = vertexIDMaxDeg(_graph[alphaID].getAdjCentersList());
    
    // Assign alpha's dominant center.
    _graph[alphaID].setDomCenter(alphaDomCenterID);
    
    // Insert alphaID into alpha's dom center's domsats.
    _graph[alphaDomCenterID].insertDomCenter(alphaID);
    
    // If alpha's degree exceeds alpha's dom center's degree.
    if ( _graph[alphaID].getDegree() > _graph[alphaDomCenterID].getDegree() ) {
      
      // Insert alpha into the priority queue.
      _graph[alphaID].setInQStatus(true);
      _priorityQ.push(_graph[alphaID]);
    }   
  }
          
  // Update using priority queue.
  fastUpdate(alphaID);
}

void OnlineStarClustering::fastUpdate(uint alphaID)
{
    uint phiID, deltaID, muID, lambdaID, vID, domCenterIDForPhi;
    //list <uint>::const_iterator iter, iter1, iter2, iter3, innerIter;
    Vertex topPriorityQ;
    
    while(!_priorityQ.empty()){
      
      topPriorityQ = _priorityQ.top();
      phiID = topPriorityQ.getID();
      _priorityQ.pop();
      
      if (_graph[phiID].isAdjCentersListEmpty()){
	
	// Set promoted to center.
	_graph[phiID].setType(Vertex::CENTER); 
        
	// CHANGE FOR FAST VERSION.            
	// Promoted to center. It will not have a dom center. 
	_graph[phiID].setDomCenterNull();
        
	// Record stats.
	//_perIterSatellitePromotions[alphaID]++;
	//_perIterNumClusters[alphaID]++;
        
	for (auto &iter: _graph[phiID].getAdjVerticesList())	  
	  _graph[iter].insertAdjCenter(phiID);
	
      }
      else {
	
	lambdaID = vertexIDMaxDeg(_graph[phiID].getAdjCentersList());
        
	if (_graph[lambdaID].getDegree() >= _graph[phiID].getDegree()){
	  
	  // If phi has a dom center then correct the dom center's list.
	  if (!_graph[phiID].isDomCenterNull()){
	    
	    domCenterIDForPhi = _graph[phiID].getDomCenter();
	    _graph[domCenterIDForPhi].deleteDomCenter(phiID);
	  }
	  
	  // If phi does not have a dom center then make lambda as its dom center.
	  _graph[phiID].setDomCenter(lambdaID);
	  _graph[lambdaID].insertDomCenter(phiID);
	}
	
	else {
	  
	  // Make phi center.
	  _graph[phiID].setType(Vertex::CENTER);
	  _graph[phiID].setDomCenterNull();
          
	  // Record stats.
	  //_perIterSatellitePromotions[alphaID]++;
	  //_perIterNumClusters[alphaID]++;
          
	  for (auto &iter2:_graph[phiID].getAdjVerticesList()){
	    _graph[iter2].insertAdjCenter(phiID);
	  }
          
	  // Get a copy, otherwise inner loop can modify the list. 
	  for (auto &iter3: _graph[phiID].getCopyOfAdjCentersList()){               
	    
	    deltaID = iter3;
	    _graph[deltaID].setType(Vertex::SATELLITE);   // Broken star.
	    _graph[deltaID].setDomCenter(phiID);    
	    _graph[phiID].insertDomCenter(deltaID); // Add deltaID to phiID's dom center list.
	    
	    //_perIterStarsBroken[alphaID]++;         // Record broken star.
	    //_perIterNumClusters[alphaID]--;         // Record dec in num of clusters.
		
	    for (auto &innerIter: _graph[deltaID].getAdjVerticesList()){
	      muID = innerIter;
	      _graph[muID].deleteAdjCenter(deltaID);
	    }                        
	    
	    for (auto &innerIter: _graph[deltaID].getDomSatsList()){
	      
	      vID = innerIter;
	      _graph[vID].setDomCenterNull();
              
	      if (_graph[vID].getInQStatus() == false){
		
		_graph[vID].setInQStatus(true);
		_priorityQ.push(_graph[vID]);
	      }
	    }                        
            
	    // Clear delta's dom centers list.
	    _graph[deltaID].clearDomCentersList();
	  }
	  
	  // Clear phi's center list.
	  _graph[phiID].clearCentersList();                
	}
      }
      
      _graph[phiID].setInQStatus(false);
      
    }// While ends. 
} // Function ends.



uint OnlineStarClustering::vertexIDMaxDeg(list<uint> const &L) const
{
  uint maxVertexID, maxVertexDeg, currVertexID, currVertexDeg;
    
  // Initialize
  currVertexID = *L.begin();
  currVertexDeg = _graph.at(currVertexID).getDegree();
    
  maxVertexID = currVertexID;
  maxVertexDeg = currVertexDeg;
    
  for (list<uint>::const_iterator it = L.begin(); it != L.end(); ++it) {
    
    currVertexID = *it;
    currVertexDeg = _graph.at(currVertexID).getDegree();
    
    if ( currVertexDeg > maxVertexDeg ){
      
      maxVertexID = currVertexID;
      maxVertexDeg = currVertexDeg;
    }
  }
  
  return maxVertexID;
}

////////////////////////////////////////////////////////
// FAST DELETE
////////////////////////////////////////////////////////
void OnlineStarClustering::deleteData(std::list<uint> ID)
{  
  for(std::list<uint>::iterator it = ID.begin(); it != ID.end(); ++it){
    fastDelete(*it);
    
    _graph.erase(*it);
    _elemInGraphSigma.remove(*it);
    _id2idx.erase(*it);

  }
  
}

void OnlineStarClustering::sortList(list <uint> &AdjCV)
{
  uint alphaID1, alphaID2, swap, i = 0;

  for(list<uint>::iterator iter1 = AdjCV.begin(); iter1 != AdjCV.end(); ++iter1){
    alphaID1 = *iter1;

    uint j = 0;

    for (list<uint>::iterator iter2 = AdjCV.begin(); j < AdjCV.size() - i - 1; ++iter2){
      alphaID2 = *iter2;

      if (_graph[alphaID1].getDegree() < _graph[alphaID2].getDegree()){
         swap = alphaID2;
         *iter2 = *iter1;
         *iter1 = swap;
      }

      j++;
    }
    i++;
  }
}

void OnlineStarClustering::fastDelete(uint alphaID){

  uint betaID, betaDomCenterID;
  uint gammaID, nuID; //DomCenterID;
  list<uint> AdjVL = _graph[alphaID].getAdjVerticesList();
  list<uint> DomSL, AdjCL;

  for (list<uint>::iterator it = AdjVL.begin(); it != AdjVL.end(); ++it){
        betaID = *it;
        _graph[betaID].decrementDegree();
        _graph[betaID].deleteAdjVertex(alphaID);

        if (!_graph[betaID].isDomCenterNull()){
          betaDomCenterID = _graph[betaID].getDomCenter();
          _graph[betaDomCenterID].getDomSatsList().remove(alphaID);
	  
          // for (list<uint>::iterator iter1 = DomSL.begin(); iter1 != DomSL.end(); ++iter1){
          //   DomCenterID = *iter1;
          //   AdjCL = _graph[DomCenterID].getAdjCentersList();
          //   sortList(AdjCL);
          // }
        } 
  }

  if(_graph[alphaID].getType() == Vertex::SATELLITE){

    AdjCL = _graph[alphaID].getAdjCentersList();

      for (list<uint>::iterator iter1 = AdjCL.begin(); iter1 != AdjCL.end(); ++iter1){
	
        betaID = *iter1;
	
        if(betaID == _graph[alphaID].getDomCenter()){
          (_graph[betaID].getDomSatsList()).remove(alphaID);
        } 

        if( !_graph[betaID].isDomSatsListEmpty() ){
            gammaID = vertexIDMaxDeg(_graph[betaID].getDomSatsList());
	    
            while(!_graph[betaID].isDomSatsListEmpty() && 
                _graph[gammaID].getDegree() > _graph[betaID].getDegree()){

                _graph[betaID].getDomSatsList().remove(gammaID);
                _graph[gammaID].setDomCenter(-1);

                if(!_graph[gammaID].getInQStatus()){
                  _graph[gammaID].setInQStatus(true);
                  _priorityQ.push(_graph[gammaID]);
                }
		
		if(_graph[betaID].getDomSatsList().size() > 0)
		  gammaID = vertexIDMaxDeg(_graph[betaID].getDomSatsList());	      
            }//end while
        }
      }//end for 
  } else {

    for(list<uint>::iterator iter1 = AdjVL.begin(); iter1 != AdjVL.end(); ++iter1){
      betaID = *iter1;
      _graph[betaID].getAdjCentersList().remove(alphaID);
    }

    DomSL = _graph[alphaID].getDomSatsList();

    if(DomSL.size() > 0){
      for(list<uint>::iterator iter1 = DomSL.begin(); iter1 != DomSL.end(); ++iter1){
	nuID = *iter1;
	_graph[nuID].setDomCenter(-1);

	if(!_graph[nuID].getInQStatus()){
	  _graph[nuID].setInQStatus(true);	          
	  _priorityQ.push(_graph[nuID]);
	}	
      }
    }

  }
  
  fastUpdate(alphaID);
}


// print out V_measure result
void OnlineStarClustering::V_measure(double beta){

  std::map<uint, std::list<uint> > C_list;  // Class list
  std::list<uint> K_list;  // Cluster list
  // bool flag = false;
  uint data_size; // Class size and data size
  //double beta = 1;

  for(map<uint,Vertex>::const_iterator it1 = _graph.begin(); 
	it1 != _graph.end(); ++it1){

    if((it1->second).getType() == Vertex::CENTER){
      K_list.push_back(it1->first);
    }

    uint label = _labels.at(it1->first);
    if( C_list.find(label) != C_list.end() ){

      C_list.at(label).push_back( it1->first );

    } else {
      std::list<uint> ls;
      ls.push_back(it1->first);
      C_list.insert(std::pair<uint, list<uint> >(label, ls));
    }
  }

  data_size = _graph.size();
  
  double H_CK = 0.0, H_C = 0.0, h;
  // homogeneity
  for(list<uint>::iterator it = K_list.begin(); it != K_list.end(); ++it){

    list<uint> ls = _graph[*it].getDomSatsList();
    uint K_size = ls.size();
    uint a_CK = 0;

    for(map<uint, list<uint>>::const_iterator it1 = C_list.begin(); it1 != C_list.end(); ++it1){
      for(list<uint>::iterator it2 = ls.begin(); it2 != ls.end(); ++it2 ){
	if(it1->first == _labels[*it2]) a_CK++;
      }

      if(a_CK != 0){
	H_CK += (double)a_CK/data_size * log10( (double)a_CK/K_size );
      }
      
      a_CK = 0;
    }
  }
  H_CK = -H_CK;

  for(map<uint, list<uint>>::const_iterator it = C_list.begin(); it != C_list.end(); ++it){
    if((it->second).size() > 0) H_C += (double)(it->second).size()/data_size * log10((double)(it->second).size()/data_size); 
  }
  H_C = -H_C;
  
  h = 1 - H_CK / H_C;

  // completeness
  double H_KC = 0.0, H_K = 0.0, c;
  for(map<uint, list<uint>>::const_iterator it = C_list.begin(); it != C_list.end(); ++it){
    uint a_CK = 0;
    for(list<uint>::const_iterator it1 = K_list.begin(); it1 != K_list.end(); ++it1){
      list<uint> ls = _graph[*it1].getDomSatsList();
      
      for(list<uint>::const_iterator it2 = ls.begin(); it2 != ls.end(); ++it2){
	if(_labels[*it2] == it->first) a_CK++;
      }

      if(a_CK != 0){
	H_KC += (double)a_CK/data_size * log10((double)a_CK/(it->second).size());
      }

      a_CK = 0;
    }
  }

  H_KC = -H_KC;

  for(list<uint>::const_iterator it = K_list.begin(); it != K_list.end(); ++it){
    uint a_CK = _graph[*it].getDomSatsList().size();
    if(a_CK != 0) H_K += (double)a_CK/data_size * log10( (double)a_CK/data_size);
  }

  H_K = -H_K;

  c = 1 - H_KC / H_K;
  
  //cout << "Homogeneity is: " << h << endl;
  //cout << "Completeness is: " << c << endl;
  cout << "V_measure is: " << (1+beta)*h*c/((beta*h)+c) << endl; 
  cout << "Number of Clusters: "<< K_list.size() << endl;
}

uint OnlineStarClustering::getDataSize()
{
  return _data.size();
}
