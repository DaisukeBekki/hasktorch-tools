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
--import Torch.Functional   (KeepDim(..),stdMeanDim)
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Tensor.Initializers    (xavierUniform')
--import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
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
      dimFF = 2048
      nHeads = 6
      nLayers = 6
      nonLinearity = Relu
      nBatches = 5
      seqLen = 13
      dimModel = nHeads * dimQK
      dropoutProb = 0
  model <- sample $ TransformerHypParams dev hasBias dimI dimQK dimFF nHeads nLayers nonLinearity
  input <- xavierUniform' dev [nBatches,seqLen,dimI]
  let pe = positionalEncoding dev seqLen dimModel (denomNumerP model)
      --att = attentionLayer model dev nHeads dimQK dropoutProb input
      enc = encoder model dev nHeads dimQK dropoutProb input
  print enc

