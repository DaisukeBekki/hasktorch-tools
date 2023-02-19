{-# LANGUAGE MultiParamTypeClasses, DeriveGeneric, RecordWildCards #-}

module Main where

import GHC.Generics               --base
import Control.Monad      (forM_) --base
--hasktorch 
import Torch.Tensor       (shape,sliceDim)
import Torch.Functional   (mseLoss)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (Parameterized(..),Randomizable(..),sample)
import Torch.Optim        (GD(..))
import Torch.Autograd     (IndependentTensor(..))
--hasktorch-tools
import Torch.Train        (update)
import Torch.Functional   (matmul)
import Torch.Tensor.TensorFactories (randnIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.LSTM   (LstmHypParams(..),InitialStatesHypParams(..),LstmParams(..),InitialStatesParams(..),lstmLayers,toDependentTensors)

data HypParams = HypParams {
    dev :: Device,
    inputDim :: Int,
    hiddenDim :: Int,
    isBiLstm ::  Bool,
    numOfLayers :: Int,
    dropout :: Maybe Double,
    projDim :: Maybe Int
    } deriving (Eq, Show)

data Params = Params {
  iParams :: InitialStatesParams,
  lParams :: LstmParams
  } deriving (Show, Generic)
instance Parameterized Params

instance Randomizable HypParams Params where
  sample HypParams{..} = Params
      <$> (sample $ InitialStatesHypParams dev isBiLstm hiddenDim numOfLayers)
      <*> (sample $ LstmHypParams dev isBiLstm inputDim hiddenDim numOfLayers True projDim)

-- | Test code to check the shapes of output tensors for the cases of
-- |   bidirectional True/False | numOfLayers 1,2,3 | dropout on/off | projDim on/off
main :: IO()
main = do
  let dev = Device CUDA 0
      iDim = 3
      hDim = 4
      numOfLayers = 5
      seqLen = 7
      isBiLstm = False
      dropout = Nothing
      projDim = Nothing
      hypParams = HypParams dev iDim hDim isBiLstm numOfLayers dropout projDim
      d = if isBiLstm then 2 else 1
      oDim = case projDim of
               Just projD -> projD
               Nothing -> hDim
  initialParams <- sample hypParams
  inputs <- randnIO' dev [seqLen,iDim]
  gt     <- randnIO' dev [seqLen,oDim]
  print $ lstmLayers (lParams initialParams) dropout (toDependentTensors $ iParams initialParams) inputs
  {-
  linearParams <- sample $ LinearHypParams dev True 5 7
  inputTensor <- randnIO' dev [2,11,5]
  let output = linearLayer linearParams inputTensor
  print output
  print $ sliceDim 1 10 11 1 output
  -}
