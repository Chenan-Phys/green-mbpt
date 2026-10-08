#include <green/gpu/thc_gpu_resident.h>
#include <green/tensors/thc_gw_fft.h>
#include <iostream>

int main(){
  try{
    using gpu=green::gpu::thc_gpu_resident;
    using matrix=gpu::matrix;
    double maximum=0;
    auto compare=[&](const matrix& a,const matrix& b){maximum=std::max(maximum,(a-b).cwiseAbs().maxCoeff());};
    auto values=[](size_t rows,size_t cols,double shift){
      matrix result(rows,cols);
      for(size_t i=0;i<rows;++i)for(size_t j=0;j<cols;++j)result(i,j)={std::sin(i*.3+j*.7+shift)*.03,std::cos(i*.4-j*.2+shift)*.02};
      return result;
    };
    for(bool low:{true,false}){
      gpu ops(low,64ul*1024*1024);
      for(auto dims:std::vector<std::pair<size_t,size_t>>{{5,9},{40,4}}){
        auto m=values(dims.first,dims.second,.1);matrix Z=m*m.adjoint();
        std::vector<matrix> response;
        for(size_t w=0;w<3;++w)response.emplace_back(values(dims.first,dims.first,.4+w));
        auto dm=ops.upload(m),dp=ops.upload(response);
        for(bool auxiliary:{false,true}){
          auto output=ops.download(ops.screen(dm,dp,auxiliary));
          for(size_t w=0;w<3;++w)compare(output[w],green::tensors::thc_screened_correlation(m,Z,response[w],false));
        }
      }
      const size_t nk=6,r=3;
      std::vector<size_t> order{5,0,4,2,1,3};std::vector<double> k,q;
      for(auto p:order){k.insert(k.end(),{double(p/3)/2+.17,double(p%3)/3+.11,.03});q.insert(q.end(),{double(p/3)/2,double(p%3)/3,0});}
      std::vector<size_t> transfers(nk*nk);
      for(size_t i=0;i<nk;++i)for(size_t j=0;j<nk;++j){
        bool found=false;
        for(size_t iq=0;iq<nk;++iq){bool same=true;
          for(size_t axis=0;axis<3;++axis){double d=k[3*j+axis]-k[3*i+axis]-q[3*iq+axis];same &= std::abs(d-std::round(d))<1e-8;}
          if(same){transfers[i*nk+j]=iq;found=true;break;}
        }
        if(!found)throw std::runtime_error("oracle transfer missing");
      }
      std::vector<matrix> x,g,a,b;
      for(size_t i=0;i<nk;++i){x.push_back(values(r,2,i));g.push_back(values(2,2,.2+i));a.push_back(values(r,r,.3+i));b.push_back(values(r,r,.6+i));}
      auto dx=ops.upload(x),dg=ops.upload(g);
      auto projected=ops.download(ops.project(dx,dg));
      for(size_t i=0;i<nk;++i)compare(projected[i],matrix(x[i]*g[i]*x[i].adjoint()));
      auto da=ops.upload(a),db=ops.upload(b);
      green::tensors::thc_momentum_fft cpu(k,q,nk);
      for(bool fft:{false,true}){
        ops.configure_momentum(nk,nk,r,transfers,k,q,fft);
        for(bool sigma:{false,true}){
          auto actual=ops.download(ops.correlate(da,db,sigma));
          auto reference=cpu.correlate(a,b,sigma,!sigma);
          for(size_t i=0;i<nk;++i)compare(actual[i],reference[i]);
        }
        std::vector<matrix> transposed=a;for(auto& v:transposed)v.transposeInPlace();
        auto actual=ops.download(ops.correlate(da,db,false,true));
        auto reference=cpu.correlate(transposed,b,false,true);
        for(size_t i=0;i<nk;++i)compare(actual[i],reference[i]);
      }
      ops.finish_stage();
    }
    bool rejected=false;
    try{gpu tiny(true,1024);tiny.allocate(16,16);}catch(const std::runtime_error&){rejected=true;}
    std::cout<<"Resident THC complex projection/screening/shifted anisotropic cuFFT error="<<maximum<<", budget_rejected="<<rejected<<std::endl;
    return maximum>1e-11 || !rejected;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}
}
