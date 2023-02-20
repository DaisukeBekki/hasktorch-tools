{-# LANGUAGE MultiParamTypeClasses, DeriveGeneric, RecordWildCards #-}

module Main where

import GHC.Generics               --base
import Control.Monad      (forM_) --base
--hasktorch 
import Torch.Tensor       (shape,sliceDim)
import Torch.Functional   (mseLoss,oneHot,embedding')
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (Parameterized(..),Randomizable(..),sample)
import Torch.Optim        (GD(..))
import Torch.Autograd     (IndependentTensor(..))
--hasktorch-tools
import Torch.Train        (update)
import Torch.Functional   (matmul)
import Torch.Tensor.TensorFactories (asTensor'',randnIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.ProtoType.Transformer (TransformerHypParams(..),TransformerParams(..),sdpAttention)

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
  m <- randnIO' dev [2,3,1,5]
  x <- randnIO' dev [2,3,5,1]
  print $ matmul m x