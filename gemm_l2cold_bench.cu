#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_timer.h"

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }while(0)
#endif
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, int(s)); exit(1);} }while(0)
#endif

// ====== オプション ======
struct Args{
  int M=4096, N=4096, K=4096;
  int trials=5;
  size_t l2_flush_bytes=0; // 0なら自動(8x L2)
  bool use_cutlass=false;
  enum { CUBLAS, ASYNC } mode = CUBLAS;
};

Args parse(int argc, char**argv){
  Args a;
  for(int i=1;i<argc;i++){
    std::string s=argv[i];
    auto nexti=[&](int&dst){ if(i+1<argc) dst=std::stoi(argv[++i]); };
    auto nextsz=[&](size_t&dst){ if(i+1<argc) dst=(size_t)std::stoll(argv[++i]); };
    if(s=="--m") nexti(a.M);
    else if(s=="--n") nexti(a.N);
    else if(s=="--k") nexti(a.K);
    else if(s=="--trials") nexti(a.trials);
    else if(s=="--mode"){ std::string v=argv[++i]; a.mode=(v=="cublas")?Args::CUBLAS:Args::ASYNC; }
    else if(s=="--flush") nextsz(a.l2_flush_bytes);
    else if(s=="--cutlass") a.use_cutlass=true;
  }
  return a;
}

// ====== L2 コールド化用バッファを大きく読む ======
__global__ void l2_flush_kernel(const uint8_t* __restrict__ buf, size_t bytes){
  // 各スレッドがストライドで全域を触る（読み捨て）
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for(size_t i=idx; i+64<=bytes; i+=stride){
    // 1行あたり64B読めば1ライン潰せる
    acc += buf[i];
  }
  if(idx==0 && acc==123456789u) printf("acc=%u\n",acc); // 最適化防止
}

void l2_flush(uint8_t* d_buf, size_t bytes, cudaStream_t stream){
  int block=256;
  int grid= (int)std::min<size_t>((bytes/64 + block-1)/block, 65535);
  l2_flush_kernel<<<grid, block, 0, stream>>>(d_buf, bytes);
}

// ====== CUDAイベントで区間測定 ======
template <class F>
float elapsed_ms(F&& work, cudaStream_t stream){
  cudaEvent_t t0,t1; CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
  CHECK_CUDA(cudaEventRecord(t0, stream));
  std::forward<F>(work)(stream);
  CHECK_CUDA(cudaEventRecord(t1, stream));
  CHECK_CUDA(cudaEventSynchronize(t1));
  float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,t0,t1));
  CHECK_CUDA(cudaEventDestroy(t0)); CHECK_CUDA(cudaEventDestroy(t1));
  return ms;
}

// ====== cublasGemmEx（Tensor Core） ======
float run_cublas(cublasHandle_t h, cudaStream_t stream,
                 int M,int N,int K, const half* dA, const half* dB, half* dC){
  CHECK_CUBLAS(cublasSetStream(h, stream));
  const float alpha=1.0f, beta=0.0f;
  auto work = [&](cudaStream_t s){
    CHECK_CUBLAS(cublasGemmEx(h,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K, // cuBLASは列メジャ前提：C = A(MxK) * B(KxN) → (N,M)で渡す
      &alpha,
      dB, CUDA_R_16F, N,
      dA, CUDA_R_16F, K,
      &beta,
      dC, CUDA_R_32F, N,
      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  return elapsed_ms(work, stream);
}

// ====== 非同期コピー版（CUTLASSを使う or 自作に差し替え） ======
#ifdef USE_CUTLASS
// CUTLASS を使う場合：-DUSE_CUTLASS と include path を通す
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
float run_async_kernel(int M,int N,int K, const half* dA,const half* dB, half* dC, cudaStream_t stream){
  using Gemm = cutlass::gemm::device::Gemm< half, cutlass::layout::RowMajor,
                                            half, cutlass::layout::RowMajor,
                                            half,  cutlass::layout::RowMajor,
                                            float,
                                            cutlass::arch::OpClassTensorOp,
                                            cutlass::arch::Sm80>; // cp.asyncを含むSM80用
  Gemm gemm_op;
  typename Gemm::Arguments args({M,N,K},
                                { (cutlass::half_t*)dA, K },
                                { (cutlass::half_t*)dB, N },
                                { dC, N },
                                { dC, N },
                                {1.0f, 0.0f});
  auto work = [&](cudaStream_t s){
    cutlass::Status st = gemm_op(args, s);
    if(st != cutlass::Status::kSuccess){ fprintf(stderr,"CUTLASS error\n"); exit(1); }
  };
  return elapsed_ms(work, stream);
}
#else
size_t initMmaAsyncStage4();
void mmaAsyncStage4(half *A, half *B, half *C, size_t M, size_t N, size_t K);

// 自作カーネルを後で入れる場合のダミー（いまは cublas と同等にしておく）
float run_async_kernel(int M,int N,int K, half* dA, half* dB, half* dC, cudaStream_t stream){
  // TODO: ここに cp.async + wmma のカーネルを実装
  // ひとまず比較が回るようにcublas版を呼ぶ（差し替えポイント）
  //cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
  //auto work = [&](cudaStream_t s){
  //  float ms = run_cublas(h, stream, M,N,K, dA,dB,dC);
  //CHECK_CUBLAS(cublasDestroy(h));
  CudaTimer timer;
  timer.start();
  mmaAsyncStage4(dA, dB, dC, M, N, K);
  return timer.end();
}
#endif

int main(int argc, char** argv){
  Args a = parse(argc, argv);
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaFree(0));

  // デバイス情報からL2サイズを取得し、フラッシュサイズを決める
  cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  size_t l2bytes = prop.l2CacheSize ? prop.l2CacheSize : 4*1024*1024; // 保険
  size_t flush_bytes = a.l2_flush_bytes ? a.l2_flush_bytes : l2bytes * 8;

  printf("GPU: %s  SM%d%d  L2=%zu KiB  flush=%zu KiB  M=%d N=%d K=%d trials=%d mode=%s\n",
    prop.name, prop.major, prop.minor, l2bytes/1024, flush_bytes/1024,
    a.M,a.N,a.K,a.trials, (a.mode==Args::CUBLAS?"cublas":"async"));

  cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));

  // メモリ確保
  size_t bytesA = (size_t)a.M*a.K*sizeof(half);
  size_t bytesB = (size_t)a.K*a.N*sizeof(half);
  size_t bytesC = (size_t)a.M*a.N*sizeof(half);
  __half *dA, *dB, *dC;
  CHECK_CUDA(cudaMalloc(&dA, bytesA));
  CHECK_CUDA(cudaMalloc(&dB, bytesB));
  CHECK_CUDA(cudaMalloc(&dC, bytesC));

  // 初期化（簡易）
  CHECK_CUDA(cudaMemsetAsync(dA, 0x3c, bytesA, stream)); // ≈ ~0.01 の半精度
  CHECK_CUDA(cudaMemsetAsync(dB, 0x3c, bytesB, stream));
  CHECK_CUDA(cudaMemsetAsync(dC, 0x00, bytesC, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // L2 フラッシュ用バッファ
  uint8_t* d_flush=nullptr; CHECK_CUDA(cudaMalloc(&d_flush, flush_bytes));

  // ウォームアップ（コンテキスト/JIT等）
  {
    cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
    (void)run_cublas(h, stream, 256,256,256, dA,dB,dC);
    CHECK_CUBLAS(cublasDestroy(h));
#ifdef USE_CUTLASS
    (void)run_async_kernel(256,256,256, dA,dB,dC, stream);
#else
    initMmaAsyncStage4(); 
    mmaAsyncStage4(dA, dB, dC, 256, 256, 256);
#endif
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // 本計測
  std::vector<float> times;
  times.reserve(a.trials);
  double flops = 2.0 * (double)a.M * (double)a.N * (double)a.K; // FMA2
  for(int t=0;t<a.trials;t++){
    // 毎回 L2 をコールド化
    l2_flush(d_flush, flush_bytes, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float ms=0;
    if(a.mode==Args::CUBLAS){
      cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
      ms = run_cublas(h, stream, a.M,a.N,a.K, dA,dB,dC);
      CHECK_CUBLAS(cublasDestroy(h));
    }else{
      ms = run_async_kernel(a.M,a.N,a.K, dA,dB,dC, stream);
    }
    times.push_back(ms);
    printf("trial %d: %.3f ms  (%.2f TFLOP/s)\n", t, ms, (float)(flops/ms/1e9));
    CHECK_CUDA(cudaMemsetAsync(dC, 0, bytesC, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // 統計
  float best=1e9, sum=0;
  for(float x: times){ best=std::min(best,x); sum+=x; }
  float avg = sum / times.size();
  printf("Best: %.3f ms  (%.2f TFLOP/s)\n", best, (float)(flops/best/1e9));
  printf("Avg : %.3f ms  (%.2f TFLOP/s)\n", avg,  (float)(flops/avg/1e9));

  cudaFree(d_flush); cudaFree(dA); cudaFree(dB); cudaFree(dC);
  cudaStreamDestroy(stream);
  return 0;
}
