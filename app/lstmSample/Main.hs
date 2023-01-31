{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where

import Control.Monad (when)
--import System.IO.Unsafe (unsafePerformIO)
import Torch.Tensor (shape,dim,sliceDim)
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
  {-
      u = asTensor'' dev [[1,2,3],[2,3,4],[5,6,7],[8,9,10]::[Float]]
      m = asTensor'' dev [[2,3],[4,5]::[Float]]
  x <- randnIO' dev [1,3]
  y <- randnIO' dev [2,3]
  z <- randnIO' dev [3]
  --when (shape u /= [2,2]) $ ioError (userError $ "illegal shape: " ++ (show $ shape u))
  print x
  print y
  print $ dim x
  print $ dim y
  print $ dim z  
  print u
  --print $ sliceDim 0 1 3 0 u
  print $ head $ shape u
  print $ sliceDim 0 0 1 1 u
  print $ shape $ sliceDim 0 0 1 1 u
  -}
  let i_size = 2
      h_size = 3
      n_layers = 4
      isBiLSTM = True
  c0h0Params <- sample $ InitialStatesHypParams dev isBiLSTM h_size n_layers
  let lstmHypParams = LstmHypParams dev isBiLSTM i_size h_size n_layers Nothing
  lstmParams <- sample lstmHypParams
  let lstm = lstmLayers lstmHypParams lstmParams (c0h0s c0h0Params)
  i <- randnIO' dev [13,2]
  lstmout <- lstm i
  gt <- randnIO' dev [13,6]
  let loss = mseLoss lstmout gt
  u <- update lstmParams GD loss 1e-1
  print u
