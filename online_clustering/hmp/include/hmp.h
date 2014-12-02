#ifndef HMP_H
#define HMP_H


namespace onlineclust{

  using uchar = unsigned char;

  
  class HMP{


  public:
    void hmp_core( uchar* rgbd);

    void pool_layer1();

    void pool_layer2();

  private:
    unsigned sparse_level;
    
    
  };

}
#endif

