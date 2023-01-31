{-# LANGUAGE DeriveGeneric #-}

module Torch.Tensor.TensorFactories (
  asTensor'',
  randIO',
  randnIO',
  randintIO',
  zeros'
) where

import Torch.Tensor (Tensor(..),TensorLike(..))
import Torch.TensorFactories (randIO,randnIO,randintIO,zeros)
import Torch.TensorOptions (defaultOpts,withDevice)
import Torch.Device (Device(..))

asTensor'' :: (TensorLike a) => Device -> a -> Tensor
asTensor'' dev v = asTensor' v $ withDevice dev defaultOpts

randIO' :: Device -> [Int] -> IO Tensor
randIO' dev shape = randIO shape $ withDevice dev defaultOpts

randnIO' :: Device -> [Int] -> IO Tensor
randnIO' dev shape = randnIO shape $ withDevice dev defaultOpts

randintIO' :: Device -> Int -> Int -> [Int] -> IO Tensor
randintIO' dev low high shape = randintIO low high shape $ withDevice dev defaultOpts

zeros' :: Device -> [Int] -> Tensor
zeros' dev shape = zeros shape $ withDevice dev defaultOpts
