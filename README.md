# hasktorch-tools

# Haskellによる深層学習

## 深層学習インターフェースとは

深層学習を用いたプログラム開発において、自動微分から実装するのでは苦労も多く、最適化もたいへんである。幸い、今日ではTensorflow, Theano, Torch, DyNet等、様々な深層学習ライブラリが開発されており、自動微分や行列計算の最適化はそれらのライブラリに任せることができるようになった。

どのライブラリにも、それぞれいくつかのプログラミング言語用のインターフェースが用意されていることが多いが、現在もっとも普及しているのはPythonインターフェースであろう。たとえばTorchを使うにはPytorch、Tensorflowを使うにはKerasがある。

Pythonは強力な行列演算ライブラリであるNumPyや、その他機械学習用のライブラリを多数備えており、深層学習向けの言語として多くの利点がある。しかし一方で、深層学習技術の開発においても（Pythonなどではなく）高度な関数型言語を使いたい、特にHaskellの静的型付けやモナドの恩恵にあずかりたい、と思うことは多々あるはずである。

## hasktorchとは

**hasktorch**は、TorchのHaskellインターフェースであり、有志による開発が進んでいる。hasktorchを利用することによって、Haskellのみを用いた深層学習プログラミングが可能になる。公式Githubはこちら[[https://github.com/hasktorch/](https://github.com/hasktorch/)] 。2021年2月現在、PyTorch 1.7に対応しており、PyTorchの更新に追随したバージョンアップも行われている。

また、hasktorchのバックエンドであるTorchの利点の一つに、GPUが使用可能という点がある。以下では、CPU環境とGPU環境において、hasktorchをセットアップする手順について解説する。CPU環境としてはDebian9環境を取り上げる。GPU環境としては、少々特殊な例ではあるが、[産総研ABCIサーバ](https://abci.ai/ja/)(CentOS6.5 with CUDA 10.1/CUDNN 7.6)を取り上げる。

Mac環境については別途記載する。

# 環境構築

## .bashrc

環境変数を設定する。以下、shellをbashと仮定して`.bashrc`での設定例を示す（shellがzshなら`.zshrc`、など各自の環境に合わせて適宜読み替えること）。最初の`HOME`,`LIB`,`PATH`,`LD_LIBRARY_PATH`の内容は各自の環境に合わせて、<username>や.local/lib等の部分を適宜書き換えること。

### CPU版

```bash
export HOME=/home/<username>
export LIB=$HOME/.local/lib
export PATH=$PATH:$HOME/.local/bin:$HOME/.local/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib
# pyenv
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# hasktorch
export USE_CUDA=0
export DEPS=$LIB/hasktorch/deps
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DEPS/pytorch/torch/lib:$DEPS/mklml/lib
```

`LD_LIBRARY_PATH`はCのダイナミックライブラリが呼び出された時に、ダイナミックライブラリを探しにいく先である。

`PYENV_ROOT`も、すでにpyenvをインストール済みであれば適宜書き換えること。今回インストールする場合は、上記のように設定しておいて以下の節に進むこと。

`USE_CUDA`は値が`0`ならばCPU環境を仮定する（GPU環境では`1`とする）。`LD_LIBRARY_PATH`に追加しているディレクトリは、hasktorch同梱のPyTorchをビルドした時に、TorchのCライブラリが生成される場所である。

### GPU版

```bash
export HOME=/home/acb11567dd
export LIB=$HOME/.local/lib
export PATH=$PATH:$HOME/.local/bin:$HOME/.local/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib
module load cuda/11.6 cudnn/8.4 gcc/9.3.0
export GCC_HOME=/apps/gcc/9.3.0/
# CUDA/CUDNNe
# export CUDA_HOME=/apps/cuda/10.2.89
export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export CUDA_INCLUDE_DIRS=$CUDA_HOME/include
export CUDA_CUDART_LIBRARY=$CUDA_HOME/lib64
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME/bin
# export CUDNN_HOME=/apps/cudnn/7.6.5/cuda10.1
export CUDNN_ROOT=$CUDNN_HOME
export CUDNN_LIBRARY=$CUDNN_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDNN_HOME/include
# pyenv
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
export CONDA=$PYENV_ROOT/versions/miniconda3-3.9.1
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# hasktorch
export USE_CUDA=1
export DEPS=$LIB/hasktorch/deps
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DEPS/pytorch/torch/lib:$CONDA/envs/pytorch/lib
export LD_PRELOAD=$CONDA/envs/pytorch/lib/libmkl_core.so:$CONDA/envs/pytorch/lib/libmkl_sequential.so
```

ABCIサーバでは、CUDA, CUDNN, GCC等は`module load`コマンドでロードする。詳しくは[公式ドキュメント](https://docs.abci.ai/ja/07/)を参照のこと。

## Pyenv setup

pytorchのインストールにはpython3環境が必要。
Pytorchの公式サイトではanaconda環境が推奨されているが、普段pyenvを使っている人は、pyenvとanacondaを両方インストールすると設定がバッティングすることが多い（要出典）。そこで、以下のようにpyenv-virtualenv環境内にminicondaをインストールする方法を勧める。Pyenv自体のインストールは [[https://scrapbox.io/bekkilab/Pyenv_install](https://scrapbox.io/bekkilab/Pyenv_install)] を参照のこと。

### CPU版

```bash
 pyenv install miniconda-3.9.1 / miniconda3-4.7.12
 pyenv shell miniconda-3.9.1
 conda create -n pytorch -c conda-forge python=3.8.1 # creates virtual environment "pytorch" under the miniconda environment
 pyenv shell miniconda-3.9.1/envs/pytorch	
 conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
 conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
```

### GPU版

```bash
 pyenv install miniconda-3.9.1
 pyenv shell miniconda-3.9.1
 conda create -n pytorch -c conda-forge python=3.8.1	# creates virtual environment "pytorch" under the miniconda environment
 pyenv shell miniconda-3.9.1/envs/pytorch
 conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses mkl-rt	# condaのモジュールがインストールできる
 conda install -c pytorch magma-cuda102 # GPU環境で必要
```

# hasktorch Download

hasktorchは[公式gitリポジトリ](https://github.com/hasktorch/hasktorch)からインストールできる。

```bash
 cd /path/to/.local/lib	# hasktorchをここの下に置く。
 git clone https://github.com/hasktorch/hasktorch.git
```

# PyTorch Install

以下の手順で、depsディレクトリ以下にpytorch 1.8.1とmklmlライブラリのバイナリがダウンロードされる。

### CPU版(conda install)

```jsx
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
```

### CPU版(install from source)

```bash
pushd /path/to/hasktorch/deps
./get-deps.sh # pytorch, mklmlをdeps以下にダウンロード。
pushd pytorch
pyenv local miniconda-3.9.1/envs/pytorch # hasktorchディレクトリ以下では常にこのpyenv環境を使う
git submodule sync
git submodule update --init --recursive
popd
```

### GPU版

```bash
pushd /path/to/hasktorch/deps
git clone --recursive https://github.com/pytorch/pytorch.git -b v1.11.0 --depth 1
cd pytorch
pyenv local minicon3-3.9.1/envs/pytorch
git submodule sync
git submodule update --init --recursive --jobs 0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

# hasktorch build

`/path/to/hasktorch/stack.yaml` （のオレンジ色の箇所）を以下のように書き換える。CPU・GPU共通。

```yaml
resolver: lts-18.14

packages:
- codegen
- libtorch-ffi
- libtorch-ffi-helper
- hasktorch
- examples
- examples/model-serving
- experimental/bounding-box
- experimental/dataloader-cifar10
- experimental/untyped-nlp

extra-include-dirs:
- deps/pytorch/torch/include
- deps/pytorch/torch/csrc/api/include
- ../../../.pyenv/versions/miniconda-3.9.1/envs/pytorch/include

extra-lib-dirs:
- deps/pytorch/torch/lib
- ../../../.pyenv/versions/miniconda-3.9.1/envs/pytorch/lib

extra-deps:
- datasets-0.4.0@sha256:9bfd5b54c6c5e1e72384a890cf29bf85a02007e0a31c98753f7d225be3c7fa6a,4929
- streaming-cassava-0.1.0.1@sha256:2d1dfeb09af62009e88311fe92f44d06dafb5cdd38879b437ea6adb3bc40acfe,1739
- streaming-bytestring-0.1.7@sha256:5b53960c1c5f8352d46a4f1a604d04340a08dcf7ff27ca3ed31253e02c01fd03,2968
- stm-2.5.0.0@sha256:c238075f9f0711cd6a78eab6001b3e218cdaa745d6377bf83cc21e58ceec2ea1,2100
- require-0.4.10@sha256:41b096daaca0d6be49239add1149af9f34c84640d09a7ffa9b45c55f089b5dac,3759
- generic-lens-2.2.0.0@sha256:4008a39f464e377130346e46062e2ac1211f9d2e256bbb1857216e889c7196be,3867
- indexed-extras-0.2@sha256:e7e498023e33016fe45467dfee3c1379862e7e6654a806a965958fa1adc00304,1349
- normaldistribution-1.1.0.3@sha256:2615b784c4112cbf6ffa0e2b55b76790290a9b9dff18a05d8c89aa374b213477,2160
- term-rewriting-0.4.0.2@sha256:5412f6aa29c5756634ee30e8df923c83ab9f012a4b8797c460af3d7078466764,2740
- type-errors-pretty-0.0.1.2@sha256:9042b64d1ac2f69aa55690576504a2397ebea8a6a55332242c88f54027c7eb57,2781
- git: https://github.com/hasktorch/tintin
  commit: 3bbe6a3797e43c92e61a2a4bdc26d5732cd5e7fd
- git: https://github.com/hasktorch/tokenizers
  commit: 9d25f0ba303193e97f8b22b9c93cbc84725886c3
  subdirs:
  - bindings/haskell/tokenizers-haskell
- git: https://github.com/hasktorch/typelevel-rewrite-rules
  commit: 1262d69e7e16705c1a85832125ce0b5b82a41278
- git: https://github.com/hasktorch/pipes-text
  commit: d4805e84327e266daa730d23982db3172f226cac
- exceptions-0.10.4@sha256:d2546046d7ba4b460d3bc7fd269cd2c52d0b7fb3cfecc038715dd0881b78a484,2796
- generic-lens-core-2.2.0.0@sha256:b6b69e992f15fa80001de737f41f2123059011a1163d6c8941ce2e3ab44f8c03,2913
- union-find-array-0.1.0.3@sha256:242e066ec516d61f262947e5794edc7bbc11fd538a0415c03ac0c01b028cfa8a,1372

allow-newer: true

nix:
  shell-file: nix/stack-shell.nix
```
## hasktorchのテスト

以下が動けばひとまずok。

```
 stack run xor-mlp
```


