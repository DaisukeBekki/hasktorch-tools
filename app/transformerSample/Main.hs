{-# LANGUAGE MultiParamTypeClasses, DeriveGeneric, RecordWildCards #-}
--{-# LANGUAGE TemplateHaskell #-}
--{-# LANGUAGE FlexibleContexts #-}
--{-# LANGUAGE FlexibleInstances #-}

module Main where

import GHC.Generics               --base
import Control.Monad      (forM_) --base
--hasktorch 
import Torch.Tensor       (shape,sliceDim)
import Torch.Functional   (Dim(..),mseLoss,transpose)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (Parameterized(..),Randomizable(..),sample)
import Torch.Optim        (GD(..))
import Torch.Autograd     (IndependentTensor(..))
--import Torch.Index        (slice)
--hasktorch-tools
import Torch.Train        (update)
import Torch.Functional   (matmul)
import Torch.Tensor.TensorFactories (asTensor'',randnIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.ProtoType.Transformer (TransformerHypParams(..),TransformerParams(..),sdpAttention)

(..>) :: (a -> b) -> (b -> c) -> a -> c
(..>) = flip fmap

main :: IO ()
main = do
  let dev = Device CPU 0
      hasBias = True
      dimI = 2
      dimQK = 3
      dimV = 4
      dimModel = 2
      nLayers = 7
      nHeads = 11
      nBatches = 5
      seqLen = 13
  model <- sample $ TransformerHypParams dev hasBias dimI dimQK dimV dimModel nHeads nLayers 
  --u <- randnIO' dev [nBatches,seqLen,dimModel]
  --v <- randnIO' dev [nBatches,seqLen,dimV]
  --print $ sdpAttention model dimQK dimV dimModel nBatches nHeads seqLen u u v
  q <- randnIO' dev [2,2,3,2]
  --k <- randnIO' dev [2,2,3,2]
  print $ q
  print $ sliceDim 2 1 2 1 q
  let f = (sliceDim 2 1 2 1) ..> (transpose (Dim 2) (Dim 3)) 
  print $ f q
  print $ matmul (sliceDim 2 1 2 1 q) q