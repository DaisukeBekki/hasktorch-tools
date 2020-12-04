#!/bin/bash

#$-l rt_F=1
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
### CUDA ###
module load cuda/10.1/10.1.243 cudnn/7.6/7.6.5 gcc/7.4.0  # 自分の必要なver.を指定しましょう
# 適宜CuDNNやNCCLも必要に応じてloadすること。
​
### Pyenv ###
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
#if which pyenv > /dev/null; then eval "$(pyenv init -)"; fi
#if which pyenv-virtualenv-init > /dev/null; then eval "$(pyenv virtualenv-init -)"; fi
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
export PYENV_VIRTUALENV_DISABLE_PROMPT=1
pyenv activate miniconda-3.9.1/envs/pytorch  # 仮想環境の有効化。pyenvならコレ
PATH=$PATH:LANG=C.UTF-8
PATH=$PATH:PYTHONIOENCODING=utf-8
PATH=$PATH:LC_ALL=ja_JP.UTF-8
​
### hasktorch ###
export USE_CUDA=1
PATH=$PATH:$LIB/hasktorch/deps/pytorch/torch/lib
PATH=$PATH:$LIB/hasktorch/libtorch/lib/
### BOOST C++ ###
export BOOST_ROOT=$LIB/boost_1_64_0
#export CPLUS_INCLUDE_PATH=$BOOST_ROOT
PATH=$PATH:$BOOST_ROOT/stage/lib
​
### actual job ###
cd hasktorch-tools
stack run regression
