#include <green/gpu/thc_gpu_resident.h>
#include <green/tensors/thc_gw_fft.h>
#include <iostream>

int main(){
  try{
    using gpu=green::gpu::thc_gpu_resident;
    using matrix=gpu::matrix;
    double maximum=0;
    auto compare=[&](const matrix& a,const matrix& b){
      if(!a.allFinite() || !b.allFinite())throw std::runtime_error("nonfinite CUDA oracle matrix");
      maximum=std::max(maximum,(a-b).cwiseAbs().maxCoeff());
    };
    auto values=[](size_t rows,size_t cols,double shift){
      matrix result(rows,cols);
      for(size_t i=0;i<rows;++i)for(size_t j=0;j<cols;++j)result(i,j)={std::sin(i*.3+j*.7+shift)*.03,std::cos(i*.4-j*.2+shift)*.02};
      return result;
    };
    for(bool aux_gemm3m:{false,true})for(bool low:{true,false}){
      gpu ops(low,64ul*1024*1024,aux_gemm3m);
      for(auto dims:std::vector<std::pair<size_t,size_t>>{{5,9},{270,4}}){
        auto m=values(dims.first,dims.second,.1);matrix Z=m*m.adjoint();
        std::vector<matrix> response;
        for(size_t w=0;w<3;++w)response.emplace_back(values(dims.first,dims.first,.4+w));
        auto dm=ops.upload(m),dp=ops.upload(response);
        for(bool auxiliary:{false,true}){
          auto output=ops.download(ops.screen(dm,dp,auxiliary));
          for(size_t w=0;w<3;++w)compare(output[w],green::tensors::thc_screened_correlation(m,Z,response[w],false));
        }
      }
      // Exercise the enlarged batched-LU boundary independently of THC data.
      for(size_t d:{64ul,150ul,256ul}){
        gpu batched(low,512ul*1024*1024,aux_gemm3m);
        std::vector<matrix> polarizations;
        for(size_t w=0;w<32;++w){matrix seed=values(d,8,.2+w);polarizations.emplace_back(-seed*seed.adjoint());}
        auto actual=batched.download(batched.screen_core(batched.upload(polarizations)));
        for(size_t w=0;w<32;++w)compare(actual[w],green::tensors::thc_screened_core(polarizations[w]));
        bool singular=false;
        try{batched.screen_core(batched.upload(std::vector<matrix>(32,matrix::Identity(d,d))));}
        catch(const std::runtime_error& e){singular=std::string(e.what()).find("singular/invalid LU")!=std::string::npos;}
        if(!singular)throw std::runtime_error("batched singular LU was not rejected");
      }
      {
        std::vector<matrix> cores;
        for(size_t q=0;q<3;++q)cores.emplace_back(values(7,5,.2+q));
        matrix response=values(7,7,.6),correlation=values(5,5,.7);
        auto dm=ops.upload(cores);
        auto compressed=ops.download(ops.compress(dm,ops.upload(response)));
        auto expanded=ops.download(ops.expand(dm,ops.upload(correlation)));
        auto dh=ops.adjoint(dm);
        auto adjoints=ops.download(dh);
        auto packed_compressed=ops.download(ops.compress(dm,ops.upload(response),dh));
        auto packed_expanded=ops.download(ops.expand(dm,ops.upload(correlation),dh));
        for(size_t q=0;q<cores.size();++q){
          compare(adjoints[q],matrix(cores[q].adjoint()));
          compare(compressed[q],matrix(cores[q].adjoint()*response*cores[q]));
          compare(expanded[q],matrix(cores[q]*correlation*cores[q].adjoint()));
          compare(packed_compressed[q],matrix(cores[q].adjoint()*response*cores[q]));
          compare(packed_expanded[q],matrix(cores[q]*correlation*cores[q].adjoint()));
        }
      }
      for(auto dims:std::vector<std::pair<size_t,size_t>>{{5,3},{3,5}}){
        const size_t r=dims.first,Q=dims.second,nt=6,nw=4;
        matrix m=values(r,Q,.1),Z=m*m.adjoint();
        matrix forward=values(nt,nw,.2),backward=values(nw,nt,.3);
        auto dm=ops.upload(m),history=ops.allocate(r,r,nt,true);
        std::vector<matrix> tau(nt),point_w(nw);
        for(size_t t=0;t<nt/2;++t){
          matrix raw=values(r,r,.4+t);tau[t]=.5*(raw+raw.adjoint()).eval();tau[nt-t-1]=tau[t];
          auto raw_device=ops.upload(raw);ops.symmetrize(raw_device);
          ops.accumulate_time(history,raw_device,t,nt,1.);ops.mirror_time(history,t,nt);
        }
        auto actual_tau=ops.download(history);
        for(size_t t=0;t<nt;++t)compare(actual_tau[t],tau[t]);
        auto compressed=ops.compress(dm,history);
        auto frequency=ops.multiply(compressed.reshape(Q*Q,nt),ops.upload(forward)).reshape(Q,Q,nw);
        auto core=ops.screen_core(frequency);
        auto core_tau=ops.multiply(core.reshape(Q*Q,nw),ops.upload(backward)).reshape(Q,Q,nt);
        auto actual=ops.download(ops.expand(dm,core_tau));
        for(size_t w=0;w<nw;++w){
          matrix response=matrix::Zero(r,r);
          for(size_t t=0;t<nt;++t)response+=forward(t,w)*tau[t];
          point_w[w]=green::tensors::thc_screened_correlation(m,Z,response,false);
        }
        for(size_t t=0;t<nt;++t){
          matrix expected=matrix::Zero(r,r);
          for(size_t w=0;w<nw;++w)expected+=backward(w,t)*point_w[w];
          compare(actual[t],expected);
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
      auto dxh=ops.adjoint(dx);
      auto packed_projection=ops.download(ops.project(dx,dg,dxh));
      auto packed_back=ops.download(ops.backproject(dx,ops.upload(a),dxh,-.7));
      for(size_t i=0;i<nk;++i){
        compare(packed_projection[i],matrix(x[i]*g[i]*x[i].adjoint()));
        compare(packed_back[i],matrix(-.7*x[i].adjoint()*a[i]*x[i]));
      }
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
        auto before=ops.fft_calls();auto prepared=ops.prepare_sigma_right(db);
        for(size_t spin=0;spin<2;++spin) {
          auto shared=ops.download(ops.correlate_sigma_prepared(da,prepared));
          auto expected=cpu.correlate(a,b,true,false);
          for(size_t i=0;i<nk;++i)compare(shared[i],expected[i]);
        }
        if(ops.fft_calls()-before!=(fft?5:0))throw std::runtime_error("CUDA screening FFT was not shared across spins");
      }
      ops.configure_momentum(nk,nk,r,transfers,k,q,false);
      auto repeated=ops.download(ops.repeat(dx,3));
      for(size_t i=0;i<3*nk;++i)compare(repeated[i],x[i%nk]);
      for(size_t iq=0;iq<nk;++iq) {
        const size_t n=2,Q=4;
        matrix m=10.*values(r,Q,.1+iq),c=10.*values(Q,Q,.3+iq);
        std::vector<matrix> strong_x=x,strong_g=g;
        for(auto& value:strong_x)value*=10.;for(auto& value:strong_g)value*=10.;
        auto vx=ops.upload(strong_x),vg=ops.upload(strong_g);
        auto vertices=ops.orbital_vertices(vx,ops.upload(m),iq);
        auto weighted=ops.multiply(vertices,ops.upload(c));
        auto actual=ops.download(ops.orbital_sigma(vertices,weighted,vg,iq));
        matrix wc=m*c*m.adjoint();
        for(size_t ik=0;ik<nk;++ik) {
          matrix expected=matrix::Zero(n,n);
          for(size_t kp=0;kp<nk;++kp)if(transfers[ik*nk+kp]==iq) {
            matrix pg=strong_x[kp]*strong_g[kp]*strong_x[kp].adjoint();
            expected+=strong_x[ik].adjoint()*pg.cwiseProduct(wc)*strong_x[ik];
          }
          compare(actual[ik],expected);
        }
      }
      ops.finish_stage();
    }
    bool rejected=false;
    {
      gpu profiled(true,4ul*1024*1024,false,true);
      auto a=profiled.upload(values(19,23,.6));
      auto ah=profiled.adjoint(a);
      auto ahh=profiled.adjoint(ah);
      compare(profiled.download(ahh)[0],values(19,23,.6));
      profiled.multiply(a,ah);profiled.finish_stage();
      auto times=profiled.component_seconds();
      if(times.empty() || !times.count("adjoint_pack") || !profiled.component_seconds().empty())
        throw std::runtime_error("CUDA component profile drain failed");
      for(const auto& item:times)if(!std::isfinite(item.second) || item.second<=0)
        throw std::runtime_error("invalid CUDA component time");
    }
    try{gpu tiny(true,1024);tiny.allocate(16,16);}catch(const std::runtime_error&){rejected=true;}
    std::cout<<"Resident THC complex projection/screening/IR core reorder/shifted anisotropic cuFFT error="<<maximum<<", budget_rejected="<<rejected<<std::endl;
    return maximum>1e-11 || !rejected;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}
}
