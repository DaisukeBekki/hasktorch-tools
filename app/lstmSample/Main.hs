{-# LANGUAGE MultiParamTypeClasses, DeriveGeneric #-}

module Main where

import Torch.Functional   (mseLoss) 
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (sample)
import Torch.Optim        (GD(..))
import Torch.Train        (update)
import Torch.Tensor.TensorFactories (randnIO')
import Torch.Layer.LSTM   (LstmHypParams(..),InitialStatesHypParams(..),lstmLayers,toDependentTensors)

main :: IO()
main = do
  let dev = Device CUDA 0
      isBiLSTM = True
      inputDim = 2
      hiddenDim = 3
      numOfLayers = 3
      projDim = 1
      seqLen = 13
      initialStatesHypParams = InitialStatesHypParams dev isBiLSTM hiddenDim numOfLayers 
      lstmHypParams = LstmHypParams dev isBiLSTM inputDim hiddenDim numOfLayers True (Just 0.5) (Just projDim)
  c0h0Params <- sample initialStatesHypParams 
  lstmParams <- sample lstmHypParams
  inputs <- randnIO' dev [seqLen,inputDim]
  gt     <- randnIO' dev [seqLen,projDim]
  let lstmOut = lstmLayers lstmHypParams lstmParams (toDependentTensors c0h0Params) inputs
      loss = mseLoss lstmOut gt
  u <- update lstmParams GD loss 5e-1
  print u
