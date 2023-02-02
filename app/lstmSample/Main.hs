{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where

import Control.Monad (when)
--import System.IO.Unsafe (unsafePerformIO)
import Torch.Tensor (shape)
import Torch.Functional (mul,squeezeAll,mseLoss)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN (sample)
import Torch.Autograd (IndependentTensor(..),makeIndependent)
import Torch.Optim        (GD(..))
import Torch.Train        (update)
import Torch.Tensor.TensorFactories (asTensor'',randnIO')
import Torch.Layer.LSTM   (LstmHypParams(..),LstmParams(..),InitialStatesHypParams(..),InitialStatesParams(..),lstmLayers)

main :: IO()
main = do
  let dev = Device CUDA 0
      i_size = 2
      h_size = 5
      seq_len = 13
      p_size = 1
      n_layers = 4
      isBiLSTM = True
  c0h0Params <- sample $ InitialStatesHypParams dev isBiLSTM h_size n_layers
  let lstmHypParams = LstmHypParams dev isBiLSTM i_size h_size n_layers Nothing (Just p_size)
  lstmParams <- sample lstmHypParams
  let lstm = lstmLayers lstmHypParams lstmParams (c0h0s c0h0Params)
  inputs <- randnIO' dev [seq_len,i_size]
  lstmout <- lstm inputs
  gt <- randnIO' dev [seq_len,p_size]
  let loss = mseLoss lstmout gt
  u <- update lstmParams GD loss 1e-1
  print u
