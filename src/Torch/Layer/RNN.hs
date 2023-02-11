{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module Torch.Layer.RNN (
  RNNHypParams(..),
  RNNParams(..),
  rnnLayer,
) where

import GHC.Generics                       --base
import System.IO.Unsafe (unsafePerformIO) --base
--hasktorch
import Torch
--hasktorch-tools
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Tensor.Util (unstack)

data RNNHypParams = RNNHypParams
  { 
    dev :: Device,
    inputDim :: Int,
    hiddenDim :: Int
  }

data RNNParams = RNNParams
  { inh :: LinearParams,
    hh :: LinearParams
  } deriving (Show, Generic, Parameterized)

instance Randomizable RNNHypParams RNNParams where
  sample RNNHypParams {..} =
    RNNParams
      <$> sample (LinearHypParams dev True inputDim hiddenDim)
      <*> sample (LinearHypParams dev True hiddenDim hiddenDim)

rnnCell ::
  RNNParams ->
  -- h_t-1
  Tensor  ->
  -- x_t
  Tensor ->
  -- h_t
  Tensor
rnnCell RNNParams {..} ht xt =
  Torch.tanh ( linearLayer inh xt + linearLayer hh ht)

rnnLayer :: RNNParams -- ^ model
  -> Tensor       -- ^ h_0 of shape [hiddenDim]
  -> Maybe Double -- ^ dropout
  -> Tensor     -- ^ input tensor of shape [seqLen, inputDim]
  -> (Tensor, Tensor) -- ^ (output, h_last) of shape [seqLen, hiddenDim] and [hiddenDim]
rnnLayer model h0 dropoutProb input =
  let dropoutLayer = case dropoutProb of
                       Just prob -> unsafePerformIO . (dropout prob True)
                       Nothing -> id
      hidden = unstack $ dropoutLayer $ stack (Dim 0) $ scanl (rnnCell model) h0 $ unstack input
  in (stack (Dim 0) $ tail hidden, last hidden)