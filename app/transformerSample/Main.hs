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
import Torch.Functional   (KeepDim(..),stdMeanDim)
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Tensor.Initializers    (xavierUniform')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.NonLinear (ActName(..))
import Torch.Layer.ProtoType.Transformer 
--(TransformerHypParams(..),TransformerParams(..),AttentionHypParams(..),AttentionParams(..),sdpAttention,positionwiseFeedForward,positionalEncoding,attentionLayer,encoder)

(.->) :: a -> (a -> b) -> b
(.->) = (&)

main :: IO ()
main = do
  let dev = Device CUDA 0
      hasBias = True
      dimI = 516
      dimQK = 1023
      nLayers = 6
      nHeads = 6
      nBatches = 5
      seqLen = 13
      dimModel = nHeads * dimQK
      dimFF = 2048
      dropoutProb = 0
  --model <- sample $ AttentionHypParams dev hasBias dimModel dimFF Relu
  model <- sample $ TransformerHypParams dev hasBias dimI dimQK dimFF nHeads nLayers Relu
  input <- xavierUniform' dev [nBatches,seqLen,dimI]
  let --pe = positionalEncoding dev seqLen dimModel
      v = encoder model dev nHeads dimQK dropoutProb input
      --u = attentionLayer model dev nHeads dimQK dropoutProb input
  --print input
  print v 
  print $ shape v
