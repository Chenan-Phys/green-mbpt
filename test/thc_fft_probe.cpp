#include <green/tensors/thc_gw_fft.h>
#include <iostream>

int main() {
  try {
    using namespace green::tensors;
    const size_t nk=6;
    std::vector<size_t> order{5,0,4,2,1,3};std::vector<double> k,q;
    for(size_t p:order){k.insert(k.end(),{double(p/3)/2+.17,double(p%3)/3+.11,.03});}
    for(size_t p:order){q.insert(q.end(),{double(p/3)/2,double(p%3)/3,0});}
    thc_momentum_fft fft(k,q,nk);
    std::vector<thc_matrix> a(nk),b(nk);
    for(size_t i=0;i<nk;++i) {
      a[i].resize(2,2);b[i].resize(2,2);
      for(size_t e=0;e<4;++e){a[i].data()[e]={std::sin(i+e*.3),std::cos(i*2+e)};b[i].data()[e]={std::cos(i+e*.7),std::sin(i+e)};}
    }
    auto correlation=fft.correlate(a,b,false,true),sigma=fft.correlate(a,b,true,false);
    auto difference=[&](size_t i,size_t j,size_t iq) {
      for(size_t axis=0;axis<3;++axis) {
        double d=k[3*i+axis]-k[3*j+axis]-q[3*iq+axis];
        if(std::abs(d-std::round(d))>1e-8)return false;
      }
      return true;
    };
    double error=0;
    for(size_t iq=0;iq<nk;++iq) {
      thc_matrix direct=thc_matrix::Zero(2,2);
      for(size_t i=0;i<nk;++i)for(size_t j=0;j<nk;++j)if(difference(i,j,iq))direct+=a[i].cwiseProduct(b[j])/double(nk);
      error=std::max(error,(direct-correlation[iq]).cwiseAbs().maxCoeff());
    }
    for(size_t i=0;i<nk;++i) {
      thc_matrix direct=thc_matrix::Zero(2,2);
      for(size_t iq=0;iq<nk;++iq)for(size_t kp=0;kp<nk;++kp)if(difference(kp,i,iq))direct+=a[kp].cwiseProduct(b[iq])/double(nk);
      error=std::max(error,(direct-sigma[i]).cwiseAbs().maxCoeff());
    }
    bool rejected=false;auto bad=k;bad[3]+=.01;
    try{thc_momentum_fft invalid(bad,q,nk);}catch(const std::exception&){rejected=true;}
    std::cout<<"THC anisotropic shuffled shifted complex correlation error="<<error<<" irregular_rejected="<<rejected<<std::endl;
    return error>1e-12 || !rejected;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}
}
