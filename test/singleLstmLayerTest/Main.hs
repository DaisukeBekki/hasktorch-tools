{-# LANGUAGE MultiParamTypeClasses, DeriveGeneric, RecordWildCards #-}

module Main where

import GHC.Generics               --base
import Control.Monad      (forM_) --base
import Test.HUnit.Base    (Test(..),(~:),assertEqual) --HUnit
import Test.HUnit.Text    (runTestTT)     --HUnit
--hasktorch 
import Torch.Tensor       (shape,sliceDim)
import Torch.Functional   (mseLoss)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (Parameterized(..),Randomizable(..),sample)
import Torch.Optim        (GD(..))
--hasktorch-tools
import Torch.Train        (update)
import Torch.Tensor.TensorFactories (randnIO')
import Torch.Layer.LSTM   (LstmHypParams(..),InitialStatesHypParams(..),LstmParams(..),InitialStatesParams(..),singleLstmLayer,toDependentTensors)

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
      iDim = 2
      hDim = 4
      isBiLstmAlt = [True]--, False]
      numOfLayersAlt = [1]--,2,3]
      dropoutAlt = [Nothing]--, Just 0.5]
      projDimAlt = [Nothing]--, Just 3]
      seqLen = 10
  forM_ (do
         x <- isBiLstmAlt
         y <- numOfLayersAlt
         z <- dropoutAlt
         w <- projDimAlt
         return (x,y,z,w)) $ \(isBiLstm, numOfLayers, dropout, projDim) -> do
    putStrLn $ "isBiLstm = " ++ (show isBiLstm)
    putStrLn $ "numOfLayers = " ++ (show numOfLayers)
    putStrLn $ "dropout = " ++ (show dropout)
    putStrLn $ "projDim = " ++ (show projDim)
    let hypParams = HypParams dev iDim hDim isBiLstm numOfLayers dropout projDim
        d = if isBiLstm then 2 else 1
        oDim = case projDim of
                 Just projD -> projD
                 Nothing -> hDim
    initialParams <- sample hypParams
    inputs <- randnIO' dev [seqLen,iDim]
    gt     <- randnIO' dev [seqLen,oDim]
    let (h0,c0) = toDependentTensors $ iParams initialParams
        h0c0 = if isBiLstm 
                 then (sliceDim 0 0 2 1 h0, sliceDim 0 0 2 1 c0)
                 else (sliceDim 0 0 1 1 h0, sliceDim 0 0 1 1 c0)
        firstLstmP = firstLstmParams $ lParams initialParams
        restLstmP = restLstmParams $ lParams initialParams
        (output, (hn, cn)) = singleLstmLayer isBiLstm hDim firstLstmP h0c0 inputs
    _ <- runTestTT $ TestList [
      "O" ~: assertEqual "shape of output" [seqLen, d * oDim] (shape output),
      "H" ~: assertEqual "shape of h_n" [d * oDim] (shape hn),
      "C" ~: assertEqual "shape of c_n" [d * hDim] (shape cn)
      ]
    putStrLn "All tests done for this case.\n"
