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
  let device = Device CUDA 0
      isBiLSTM = True
      i_size = 2
      h_size = 3
      n_layers = 3
      p_size = 1
      seq_len = 13
      initialStatesHypParams = InitialStatesHypParams {
        dev' = device
        , bidirectional' = isBiLSTM
        , hidden_size' = h_size
        , num_layers' = n_layers
        }
      lstmHypParams = LstmHypParams {
        dev = device
        , bidirectional = isBiLSTM
        , input_size    = i_size
        , hidden_size   = h_size
        , num_layers    = n_layers
        , dropoutProb   = Just 0.5
        , proj_size     = Just p_size
        }
  c0h0Params <- sample initialStatesHypParams 
  lstmParams <- sample lstmHypParams
  inputs <- randnIO' device [seq_len,i_size]
  gt     <- randnIO' device [seq_len,p_size]
  let lstm = lstmLayers lstmHypParams lstmParams (c0h0s c0h0Params)
  let loss = mseLoss (lstm inputs) gt
  u <- update lstmParams GD loss 5e-1
  print u
