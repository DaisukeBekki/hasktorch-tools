{-# LANGUAGE DeriveGeneric, RecordWildCards, MultiParamTypeClasses,FlexibleInstances  #-}

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
import Torch.Functional   (squeezeAll)
import Torch.Device       (Device(..))
import Torch.NN           (Parameterized,Randomizable,sample)
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.NonLinear (ActName(..),decodeAct)

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
    let layersSpecs = (inputDim,Id):layerSpecs 
    layers <- forM (toPairwise layersSpecs) $ \((iDim,_),(outputDim,outputAct)) -> do
          linearL <- sample $ LinearHypParams dev True iDim outputDim
          return $ (linearL, decodeAct outputAct)
    return $ MLPParams layers

{-
instance Show MLPParams where
  show MLPParams{..} =
    "Input Layer:\n" ++ "\nOutput Layer:\n"
-}

mlpLayer :: MLPParams -> Tensor -> Tensor -- squeezeALlするのでスカラーが返る
mlpLayer MLPParams{..} input = squeezeAll $ foldl' (\vec (layerParam, act) -> act $ linearLayer layerParam vec) input layers

-- | Example:
-- | toPairwise [(4,"a"),(5,"b"),(6,"c")] = [((4,"a"),(5,"b")),((5,"b"),(6,"c"))]
toPairwise :: [a] -> [(a,a)]
toPairwise [] = []
toPairwise [_] = []
toPairwise (x : (y : xs)) =
  scanl shift (x, y) xs
  where
    shift (_, p) q = (p, q)

