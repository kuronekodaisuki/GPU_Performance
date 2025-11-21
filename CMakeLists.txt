cmake_minimum_required(VERSION 3.24)  # CUDA_ARCHITECTURES の native/all が使える世代
project(GPU_Performance LANGUAGES CXX CUDA)

# --- C++ / CUDA 標準 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- CUDA Toolkit を検出（nvcc / include / libs）---
find_package(CUDAToolkit REQUIRED)  # CUDA::cudart, CUDA::cublas などのimported targetが用意される
#  ^ CMake公式の推奨は FindCUDA(旧) ではなく FindCUDAToolkit(新) :contentReference[oaicite:1]{index=1}

# --- 実行ファイル ---
add_executable(bench
    gemm_l2cold_bench.cu
    3rd_party/mma_async_stage4.cu
)

# --- CUDAアーキ設定 ---
# 例1: 手元のGPUに合わせて自動（CMake 3.24+）
set_property(TARGET bench PROPERTY CUDA_ARCHITECTURES native)
# 例2: 明示指定したい場合
# set_property(TARGET bench PROPERTY CUDA_ARCHITECTURES 75 86)

# CUDA_ARCHITECTURES は空だとエラーになるので必ず設定するのが安全 :contentReference[oaicite:2]{index=2}

# --- include dirs ---
# cuda_fp16.h / cublas_v2.h は CUDA include にあるので、CUDAToolkitが持つincludeを追加
target_include_directories(bench PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    3rd_party/common
)
# もしくは target_link_libraries だけでも include が伝播する構成が多いですが、
# host側 .cpp で cuda_fp16.h を読むときは明示追加が確実 :contentReference[oaicite:3]{index=3}

# --- link ---
target_link_libraries(bench PRIVATE
    CUDA::cudart
    CUDA::cublas
)

# --- CUDAコンパイルオプション（halfやlambda等を使う場合に便利）---
target_compile_options(bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

# --- separable compilation が必要ならON（複数cuに跨るdevice呼び出し等）---
set_target_properties(bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# --- 便利な定義例（必要なら） ---
# target_compile_definitions(bench PRIVATE CUDA_FP16_ENABLED=1)
