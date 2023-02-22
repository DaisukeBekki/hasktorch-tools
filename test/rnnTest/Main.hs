{-# LANGUAGE MultiParamTypeClasses, DeriveGeneric, RecordWildCards #-}

module Main where

import GHC.Generics               --base
import Control.Monad      (forM_) --base
import Test.HUnit.Base    (Test(..),(~:),assertEqual) --HUnit
import Test.HUnit.Text    (runTestTT)     --HUnit
--hasktorch 
import Torch.Tensor       (shape)
import Torch.Functional   (mseLoss)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (Parameterized(..),Randomizable(..),sample)
import Torch.Optim        (GD(..))
import Torch.Autograd     (IndependentTensor(..))
--hasktorch-tools
import Torch.Train        (update)
import Torch.Tensor.TensorFactories (randnIO')
import Torch.Layer.RNN   (RnnHypParams(..),InitialStatesHypParams(..),RnnParams(..),InitialStatesParams(..),rnnLayers)
import Torch.Layer.NonLinear (ActName(..))

data HypParams = HypParams {
    dev :: Device,
    inputDim :: Int,
    hiddenDim :: Int,
    isBidirectional ::  Bool,
    hasBias :: Bool,
    numOfLayers :: Int,
    dropout :: Maybe Double
    } deriving (Eq, Show)

data Params = Params {
  iParams :: InitialStatesParams,
  rParams :: RnnParams
  } deriving (Show, Generic)
instance Parameterized Params

instance Randomizable HypParams Params where
  sample HypParams{..} = Params
      <$> (sample $ InitialStatesHypParams dev isBidirectional hiddenDim numOfLayers)
      <*> (sample $ RnnHypParams dev isBidirectional inputDim hiddenDim numOfLayers True)

-- | Test code to check the shapes of output tensors for the cases of
-- |   bidirectional True/False | numOfLayers 1,2,3 | dropout on/off | projDim on/off
main :: IO()
main = do
  let dev = Device CUDA 0
      iDim = 2
      hDim = 5
      isBidirectionalAlt = [True, False]
      hasBiasAlt = [True, False]
      numOfLayersAlt = [1,2,3]
      dropoutAlt = [Nothing, Just 0.5]
      actNameAlt = [Sigmoid,Tanh]--,Relu,Elu,Selu,Id]
      seqLen = 11
  forM_ (do
         x <- isBidirectionalAlt
         y <- hasBiasAlt
         z <- numOfLayersAlt
         u <- dropoutAlt
         a <- actNameAlt
         return (x,y,z,u,a)) $ \(isBidirectional, hasBias, numOfLayers, dropout, actName) -> do
    putStr $ "Setting: isBidirectional = " ++ (show isBidirectional)
    putStr $ " / hasBias = " ++ (show hasBias)
    putStr $ " / numOfLayers = " ++ (show numOfLayers)
    putStr $ " / dropout = " ++ (show dropout)
    putStr $ " / actName = " ++ (show actName) ++ "\n\n"
    let hypParams = HypParams dev iDim hDim isBidirectional hasBias numOfLayers dropout
        d = if isBidirectional then 2 else 1
    initialParams <- sample hypParams
    inputs <- randnIO' dev [seqLen,iDim]
    gt     <- randnIO' dev [seqLen,d * hDim]
    let rnnOut = fst $ rnnLayers (rParams initialParams) actName dropout (toDependent $ h0s $ iParams initialParams) inputs
        loss = mseLoss rnnOut gt
    (u,_) <- update initialParams GD loss 5e-1
    let (output,hn) = rnnLayers (rParams u) actName dropout (toDependent $ h0s $ iParams u) inputs
    _ <- runTestTT $ TestList [
      "A" ~: assertEqual "shape of output" [seqLen, d * hDim] (shape output),
      "B" ~: assertEqual "shape of h_n" [d * numOfLayers, hDim] (shape hn)
      ]
    putStrLn "Case clear.\n"
