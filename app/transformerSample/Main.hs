{-# LANGUAGE MultiParamTypeClasses, DeriveGeneric, RecordWildCards #-}
--{-# LANGUAGE TemplateHaskell #-}
--{-# LANGUAGE FlexibleContexts #-}
--{-# LANGUAGE FlexibleInstances #-}

module Main where

import GHC.Generics               --base
import Control.Monad      (forM_) --base
import Data.Function      ((&))   --base
--hasktorch 
import Torch.Tensor       (shape,reshape,sliceDim)
import Torch.Functional   (Dim(..),mseLoss,transpose)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (Parameterized(..),Randomizable(..),sample)
import Torch.Optim        (GD(..))
import Torch.Autograd     (IndependentTensor(..))
--import Torch.Index        (slice)
--hasktorch-tools
import Torch.Train        (update)
import Torch.Functional   (matmul)
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Tensor.Initializers    (xavierUniform')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.NonLinear (ActName(..))
import Torch.Layer.ProtoType.Transformer (TransformerHypParams(..),TransformerParams(..),sdpAttention,positionwiseFeedForward,positionalEncoding)

(.->) :: a -> (a -> b) -> b
(.->) = (&)

main :: IO ()
main = do
  let dev = Device CPU 0
      hasBias = True
      dimI = 2
      dimQK = 3
      nLayers = 3
      nHeads = 5
      nBatches = 5
      seqLen = 11
      dimModel = nHeads * dimQK
      dimFF = 10
  model <- sample $ TransformerHypParams dev hasBias dimI dimQK nHeads nLayers (0.1) [(dimFF,Relu)]
  q <- xavierUniform' dev [nBatches,seqLen,dimModel]
  k <- xavierUniform' dev [nBatches,seqLen,dimModel]
  v <- xavierUniform' dev [nBatches,seqLen,dimModel]
  let x = sdpAttention dev nBatches nHeads seqLen dimQK q k v
      y = positionwiseFeedForward 
      z = positionalEncoding dev seqLen dimModel 
  print z