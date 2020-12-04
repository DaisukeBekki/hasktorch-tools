#      - hasktorch-toolsTHを復元すること 例えば $HOME/local/bin とか。
export HOME=/home/acc12247eu
export LIB=$HOME/.local/lib
PATH=$PATH:$HOME/.local/bin
​
### CUDA ###
#PATH=$PATH:/apps/gcc/7.4.0/bin/gcc
. /etc/profile.d/modules.sh
module load cuda/10.1/10.1.243 cudnn/7.6/7.6.5 gcc/7.4.0  # 自分の必要なver.を指定しましょう
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB/hasktorch/deps/pytorch/torch/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BOOST_ROOT/stage/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB/hasctorch/libtorch/lib/
#PATH=$PATH:$LIB/hasktorch/deps/pytorch/torch/lib
#PATH=$PATH:$LIB/hasktorch/libtorch/lib/
### BOOST C++ ###
#export BOOST_ROOT=$LIB/boost_1_64_0
#export CPLUS_INCLUDE_PATH=$BOOST_ROOT
#PATH=$PATH:$BOOST_ROOT/stage/lib
​
### actual job ###
cd work/hasktorch-tools
stack run seq-reg-train

