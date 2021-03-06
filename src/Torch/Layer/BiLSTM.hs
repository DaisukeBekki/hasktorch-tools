{-# LANGUAGE DeriveGeneric #-}

module Torch.Layer.BiLSTM (
  BiLstmHypParams(..),
  BiLstmParams(..),
  biLstmLayers
  ) where

import Prelude hiding (tanh) 
import GHC.Generics              --base
import Data.List (foldl') --base
import Control.Monad (forM)      --base
--hasktorch
import Torch.Tensor       (Tensor(..))
import Torch.Device       (Device(..))
import Torch.NN           (Parameterized,Randomizable,sample)
import Torch.Layer.LSTM   (LstmHypParams(..),LstmParams(..),biLstmLayer)

data BiLstmHypParams = BiLstmHypParams {
  dev :: Device,
  numOfLayers :: Int,
  stateDim :: Int
  } deriving (Eq, Show)

data BiLstmParams = BiLstmParams {
  gates :: [LstmParams]
  } deriving (Show, Generic)

instance Parameterized BiLstmParams

instance Randomizable BiLstmHypParams BiLstmParams where
  sample BiLstmHypParams{..} = 
    BiLstmParams <$>
      forM [1..numOfLayers] (\_ -> sample $ LstmHypParams dev stateDim)

biLstmLayers :: BiLstmParams -> (Tensor,Tensor) -> [Tensor] -> [(Tensor,Tensor)]
biLstmLayers BiLstmParams{..} (c0,h0) inputs =
  let lstms = map biLstmLayer gates;
      firstLayer = (head lstms) (c0,h0) inputs in
  foldl' (\lstm newLayer -> newLayer (head lstm) (snd $ unzip lstm)) firstLayer (tail lstms) 
 
