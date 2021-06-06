{-# LANGUAGE DeriveGeneric, RecordWildCards, MultiParamTypeClasses  #-}

module Torch.Layer.MLP (
  MLPHypParams(..),
  MLPParams(..),
  ActName(..),
  mlpLayer
  ) where

import Prelude hiding (tanh)
import Control.Monad (forM) --base
import Data.List (foldl') --base
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
  layerSpecs :: [(Int,ActName)]
  } deriving (Eq, Show)

-- | DeriveGeneric Pragmaが必要
data MLPParams = MLPParams {
  layers :: [(LinearParams, Tensor -> Tensor)]
  } deriving (Generic)

instance Parameterized MLPParams

instance Randomizable MLPHypParams MLPParams where
  sample MLPHypParams{..} = do
    let layersSpecs = (inputDim,Sigmoid):layerSpecs 
    layers <- forM (toPairwise layersSpecs) $ \((inputDim,_),(outputDim,outputAct)) -> do
          linearLayer <- sample $ LinearHypParams dev inputDim outputDim
          return $ (linearLayer, decode outputAct)
    return $ MLPParams layers

{-
instance Show MLPParams where
  show MLPParams{..} =
    "Input Layer:\n" ++ "\nOutput Layer:\n"
-}

mlpLayer :: MLPParams -> Tensor -> Tensor -- squeezeALlするのでスカラーが返る
mlpLayer MLPParams{..} input = squeezeAll $ foldl' (\vec (layerParam, act) -> act $ linearLayer layerParam vec) input layers
--mlpLayer MLPParams{..} = squeezeAll . a2 . linearLayer l2 . a1 . linearLayer l1

-- | Example:
-- | toPairwise [(4,"a"),(5,"b"),(6,"c")] = [((4,"a"),(5,"b")),((5,"b"),(6,"c"))]
toPairwise :: [a] -> [(a,a)] 
toPairwise (a : (b : t)) =
  scanl shift (a, b) t
  where
    shift (a, b) c = (b, c)

