{-# LANGUAGE DeriveGeneric #-}

module Torch.Layer.MLP (
  MLPHypParams(..),
  MLPParams(..),
  ActName(..),
  mlpLayer
  ) where

import Prelude hiding (tanh)
import GHC.Generics       --base
import Torch.Tensor       (Tensor(..))
import Torch.Functional   (sigmoid,tanh,relu,selu,squeezeAll)
import Torch.Device       (Device(..))
import Torch.NN           (Parameterized,Randomizable,sample)
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)

data ActName = Sigmoid | Tanh | Relu | Selu deriving (Eq,Show)

decode :: ActName -> Tensor -> Tensor
decode actname = case actname of
                   Sigmoid -> sigmoid
                   Tanh -> tanh
                   Relu -> relu
                   Selu -> selu

data MLPHypParams = MLPHypParams {
  dev :: Device,
  inputDim :: Int,
  hiddenDim :: Int,
  outputDim :: Int,
  act1 :: ActName,
  act2 :: ActName
  } 

-- | DeriveGeneric Pragmaが必要
data MLPParams = MLPParams {
  l1 :: LinearParams,
  l2 :: LinearParams,
  a1 :: Tensor -> Tensor,
  a2 :: Tensor -> Tensor
  } deriving (Generic)

instance Parameterized MLPParams

instance Randomizable MLPHypParams MLPParams where
  sample MLPHypParams{..} = 
    MLPParams
    <$> sample (LinearHypParams dev inputDim hiddenDim)
    <*> sample (LinearHypParams dev hiddenDim outputDim)
    <*> return (decode act1)
    <*> return (decode act2)

instance Show MLPParams where
  show MLPParams{..} =
    "Input Layer:\n"
    ++ (show l1)
    ++ "\nOutput Layer:\n"
    ++ (show l2)

mlpLayer :: MLPParams -> Tensor -> Tensor -- squeezeALlするのでスカラーが返る
mlpLayer MLPParams{..} = squeezeAll . a2 . linearLayer l2 . a1 . linearLayer l1
