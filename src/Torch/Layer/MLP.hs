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
  } deriving (Eq, Show)

-- | DeriveGeneric Pragmaが必要
data MLPParams = MLPParams {
  l1 :: LinearParams,
  l2 :: LinearParams
  } deriving (Generic)

instance Parameterized MLPParams

instance Randomizable MLPHypParams MLPParams where
  sample MLPHypParams{..} = 
    MLPParams
    <$> sample (LinearHypParams dev inputDim hiddenDim)
    <*> sample (LinearHypParams dev hiddenDim outputDim)

instance Show MLPParams where
  show MLPParams{..} =
    "Input Layer:\n" ++ "\nOutput Layer:\n"

mlpLayer :: MLPHypParams -> MLPParams -> Tensor -> Tensor -- squeezeALlするのでスカラーが返る
mlpLayer MLPHypParams{..} MLPParams{..} = squeezeAll . (decode act2) . linearLayer l2 . (decode act1) . linearLayer l1
