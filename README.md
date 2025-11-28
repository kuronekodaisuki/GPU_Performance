Tensor Coreの最適化実験。

非同期コピーなどを駆使して行列積を最適化する実験です。

#git cloneの後

>mkdir build

>cd build

>cmake ..

>make

>./bench --mode cublas

>./bench --mode async

