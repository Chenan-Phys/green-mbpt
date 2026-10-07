#include <green/mbpt/common_defs.h>
#include <green/mbpt/df_integral_t.h>
#include <green/gpu/df_integral_t.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_utils.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>

int main(int argc,char** argv) {
  MPI_Init(&argc,&argv); green::utils::context();
  int result=0;
  try {
    if(argc!=4) throw std::runtime_error("usage: thc_reader_probe input.h5 thc_dir expanded_dir");
    using namespace green::mbpt;
    green::params::params p("THC original-Q pair provider test");
    green::symmetry::define_parameters(p); green::grids::define_parameters(p); define_parameters(p);
    p.parse(std::string("probe --input_file ")+argv[1]);
    green::symmetry::brillouin_zone_utils bz(p);
    size_t n,Q; green::h5pp::archive input(argv[1]); input["params/nao"]>>n; input["params/NQ"]>>Q; input.close();
    green::integrals::thc_reader_options options; options.enabled=true; options.input_file=argv[1];
    df_integral_t fitted(argv[2],n,Q,bz,green::utils::context(),options), expanded(argv[3],n,Q,bz);
    green::gpu::df_integral_t gpu(argv[2],n,bz.nk(),Q,bz,options);
    ztensor<3> a(Q,n,n),b(Q,n,n),c(Q,n,n),slice(Q-1,n,n);
    double cpu_error=0,gpu_error=0,slice_error=0;
    for(size_t i=0;i<bz.nk();++i) for(size_t j=0;j<bz.nk();++j) {
      fitted.read_integrals(i,j); expanded.read_integrals(i,j); gpu.read_integrals(i,j);
      fitted.symmetrize(a,i,j); expanded.symmetrize(b,i,j); gpu.symmetrize(c,i,j);
      fitted.symmetrize(slice,i,j,1,Q-1);
      for(size_t x=0;x<a.size();++x) {cpu_error=std::max(cpu_error,std::abs(a.data()[x]-b.data()[x]));gpu_error=std::max(gpu_error,std::abs(a.data()[x]-c.data()[x]));}
      for(size_t x=0;x<slice.size();++x) slice_error=std::max(slice_error,std::abs(slice.data()[x]-a.data()[x+n*n]));
    }
    if(!green::utils::context().global_rank) std::cout<<std::setprecision(17)<<"{\"cpu_expanded_max\":"<<cpu_error<<",\"gpu_host_cpu_max\":"<<gpu_error<<",\"original_Q_slice_max\":"<<slice_error<<"}"<<std::endl;
    if(cpu_error>1e-8 || gpu_error>1e-12 || slice_error>1e-12) result=1;
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl; result=2;}
  MPI_Finalize(); return result;
}
