{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module Torch.Layer.RNN (
  RNNHypParams(..),
  RNNParams(..),
  rnnLayer,
) where

import Torch
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import GHC.Generics

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

rnnLayer ::
  -- | model
  RNNParams ->
  -- | h_0
  Tensor ->
  -- | input
  [Tensor] -> 
  -- | (output, h_last)
  ([Tensor], Tensor)
rnnLayer model h0 input =
  let hidden = scanl (rnnCell model) h0 input
  in (tail hidden, last hidden)