#include <green/tensors/thc_gw_fft.h>
#include <green/tensors/thc_sigma_orbital.h>
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
    {
      thc_momentum_fft shared(k,q,nk);
      auto right=shared.prepare_right(b);
      for(size_t spin=0;spin<2;++spin) {
        auto result=shared.correlate_prepared(a,right);
        for(size_t i=0;i<nk;++i)error=std::max(error,(result[i]-sigma[i]).cwiseAbs().maxCoeff());
      }
      if(shared.fft_calls()!=5)throw std::runtime_error("screening FFT was not shared across spins");
    }
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
    // An independent point-space Dyson solve checks the deferred auxiliary
    // expansion for complex rectangular M and complex linear time transforms.
    // Include Q>r as well as Q<r so this does not rely on a square/real core.
    for(auto dims:std::vector<std::pair<size_t,size_t>>{{5,3},{3,5}}) {
      const size_t r=dims.first,Q=dims.second,nt=6,nw=4;
      auto values=[](size_t rows,size_t cols,double shift) {
        thc_matrix result(rows,cols);
        for(size_t i=0;i<rows;++i)for(size_t j=0;j<cols;++j)
          result(i,j)={.03*std::sin(.3*i+.7*j+shift),.02*std::cos(.4*i-.2*j+shift)};
        return result;
      };
      thc_matrix m=values(r,Q,.1),Z=m*m.adjoint(),forward=values(nw,nt,.2),backward=values(nt,nw,.3);
      {
        const size_t n=4;
        thc_matrix x=10.*values(r,n,.7),xp=10.*values(r,n,.9),g=10.*values(n,n,.2),c=10.*values(Q,Q,.8);
        auto v=thc_orbital_vertex(x,xp,m);
        thc_matrix weighted=v*c;
        thc_matrix actual=thc_orbital_sigma_weighted(v,weighted,g);
        thc_matrix wc=m*c*m.adjoint(),pg=xp*g*xp.adjoint();
        thc_matrix expected=x.adjoint()*pg.cwiseProduct(wc)*x;
        if(!actual.allFinite() || !expected.allFinite())throw std::runtime_error("nonfinite orbital Sigma oracle");
        error=std::max(error,(actual-expected).cwiseAbs().maxCoeff());
      }
      std::vector<thc_matrix> tau(nt),compressed(nt),point_w(nw),core_w(nw);
      for(size_t t=0;t<nt/2;++t) {
        thc_matrix raw=values(r,r,.4+t),P=m.adjoint()*raw*m;
        tau[t]=.5*(raw+raw.adjoint()).eval();
        compressed[t]=.5*(P+P.adjoint()).eval();
        tau[nt-t-1]=tau[t];compressed[nt-t-1]=compressed[t];
        thc_matrix expected=m.adjoint()*tau[t]*m;
        if(!compressed[t].allFinite() || !expected.allFinite())throw std::runtime_error("nonfinite compressed IR oracle");
        error=std::max(error,(compressed[t]-expected).cwiseAbs().maxCoeff());
      }
      for(size_t w=0;w<nw;++w) {
        thc_matrix response=thc_matrix::Zero(r,r),P=thc_matrix::Zero(Q,Q);
        for(size_t t=0;t<nt;++t){response+=forward(w,t)*tau[t];P+=forward(w,t)*compressed[t];}
        point_w[w]=thc_screened_correlation(m,Z,response,false);
        core_w[w]=thc_screened_core(P);
      }
      for(size_t t=0;t<nt;++t) {
        thc_matrix point_tau=thc_matrix::Zero(r,r),core_tau=thc_matrix::Zero(Q,Q);
        for(size_t w=0;w<nw;++w){point_tau+=backward(t,w)*point_w[w];core_tau+=backward(t,w)*core_w[w];}
        thc_matrix expected=m*core_tau*m.adjoint();
        if(!point_tau.allFinite() || !expected.allFinite())throw std::runtime_error("nonfinite expanded IR oracle");
        error=std::max(error,(point_tau-expected).cwiseAbs().maxCoeff());
      }
    }
    bool rejected=false;auto bad=k;bad[3]+=.01;
    try{thc_momentum_fft invalid(bad,q,nk);}catch(const std::exception&){rejected=true;}
    std::cout<<"THC anisotropic shuffled shifted correlation/complex IR core reorder error="<<error<<" irregular_rejected="<<rejected<<std::endl;
    return error>1e-12 || !rejected;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}
}
