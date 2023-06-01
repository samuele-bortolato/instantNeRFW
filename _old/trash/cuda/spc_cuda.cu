#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <tuple>

__device__ int Morton_3D_Encode_10bit( int index1, int index2, int index3 )
    { // pack 3 10-bit indices into a 30-bit Morton code
      index1 &= 0x000003ff;
      index2 &= 0x000003ff;
      index3 &= 0x000003ff;
      index1 |= ( index1 << 16 );
      index2 |= ( index2 << 16 );
      index3 |= ( index3 << 16 );
      index1 &= 0x030000ff;
      index2 &= 0x030000ff;
      index3 &= 0x030000ff;
      index1 |= ( index1 << 8 );
      index2 |= ( index2 << 8 );
      index3 |= ( index3 << 8 );
      index1 &= 0x0300f00f;
      index2 &= 0x0300f00f;
      index3 &= 0x0300f00f;
      index1 |= ( index1 << 4 );
      index2 |= ( index2 << 4 );
      index3 |= ( index3 << 4 );
      index1 &= 0x030c30c3;
      index2 &= 0x030c30c3;
      index3 &= 0x030c30c3;
      index1 |= ( index1 << 2 );
      index2 |= ( index2 << 2 );
      index3 |= ( index3 << 2 );
      index1 &= 0x09249249;
      index2 &= 0x09249249;
      index3 &= 0x09249249;
      return( index1 | ( index2 << 1 ) | ( index3 << 2 ) );
    }


__global__ void convert_to_morton(const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> quantized_points,
                                    uint8_t *temp,
                                    int begin,
                                    int end){

    int thid = threadIdx.x;

    if(thid==0){
        int t=temp[0]*0+1;
        temp[0] = t;
    }
    // if(thid < end ){
    //     int idx = Morton_3D_Encode_10bit(quantized_points[thid][0],quantized_points[thid][1],quantized_points[thid][2]);
    //     temp[thid] = 1;
    // }
}


__global__ void up_sweep(torch::PackedTensorAccessor32<uint8_t,1> temp,
                            int begin0,
                            int end0,
                            int begin1,
                            int end1
                            ){

    int thid = blockIdx.x*blockDim.x + threadIdx.x;

    if(thid < end0-begin0){

        uint8_t value = 0;
        for(int i=7; i>=0; i--){
            value = value << 1;
            if(temp[begin1 + thid*8 + i] > 0){
                value += 1;
            }
        }
        temp[begin0 + thid] = value;
    }
}


__global__ void down_sweep(torch::PackedTensorAccessor32<uint8_t,1> temp,
                                int *begin,
                                int *end
                                ){
                                    
                                }



#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
  }                                                         \
}

std::tuple<std::vector<torch::Tensor>,const char*> spc_cuda( torch::Tensor quantized_points, int levels) {

    int begin[9];
    int end[9];

    int tot_len = 0;
    for(int i=0; i<levels+1; i++){
        int len = pow(pow(2,i),3);
        begin[i] = tot_len;
        tot_len += len;
        end[i] = tot_len;
    }
    
    uint8_t *temp; 
    cudaMalloc((void **)&temp, tot_len*sizeof(uint8_t));

    const int threads = 512;

    //convert data to z-curve
    int blocks = (quantized_points.sizes()[0]+threads-1) / threads;

    

    convert_to_morton<<<blocks, threads>>>(quantized_points.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                        temp,
                                        begin[levels],
                                        end[levels]);

    cudaDeviceSynchronize();
    const char *result;
    cudaError_t cucheck_err = cudaGetLastError();
    if(cucheck_err != cudaSuccess) {
        result = cudaGetErrorString(cucheck_err);
        // assert(0);
    }else{
        result = "Correct";
    }
    
    // //compute spc values
    // for(int level=(levels-1); level>=0; level--){
    //     int elems = pow(pow(2,level),3);
    //     int blocks = (elems+threads-1) / threads;
    //     up_sweep<<<blocks, threads>>>(temp.packed_accessor32<uint8_t,1>(),
    //                                     begin[level],
    //                                     end[level],
    //                                     begin[level+1],
    //                                     end[level+1]);
    // }


    //compress spc

    auto opt1 = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor b = torch::zeros({9}, opt1);
    torch::Tensor e = torch::zeros({9}, opt1);
    for(int i=0; i<9; i++){
        b[i] = begin[i];
        e[i] = end[i];
    }

    torch::Tensor o= torch::zeros({9}, opt1);
    o[0] = quantized_points.sizes()[0];
    o[1] = blocks;
    o[2] = begin[levels];
    o[3] = end[levels];

    torch::Tensor t= torch::from_blob(temp, {tot_len});

    return {{t,b,e,o},result};

}
