{-# LANGUAGE DeriveGeneric, DisambiguateRecordFields #-}

module Torch.Layer.ProtoType.Transformer (
  TransformerHypParams(..)
  , sdpAttention
  ) where 

import GHC.Generics              --base
--hasktorch
import Torch.Tensor       (Tensor(..))
import Torch.Device       (Device(..))
import Torch.Functional   (Dim(..),matmul,transpose,softmax)
--hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'')

data TransformerHypParams = LstmHypParams {
  dev :: Device
  , inputDim :: Int  -- ^ The number of expected features in the input x
  , hasBias :: Bool  -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  } deriving (Eq, Show)

-- | scaled dot-product attention
sdpAttention :: Device -- ^ device
  -> Int -- ^ key dim (of Q,K)
  -> Int -- ^ value dim (of V)
  -> Int -- ^ number of heads (of Q,K,V)
  -> Int -- ^ seq length
  -> Tensor -- ^ Q of shape [seq,dh,dk]
  -> Tensor -- ^ K of shape [seq,dh,dk] 
  -> Tensor -- ^ V of shape [seq,dh,dv]
  -> Tensor
sdpAttention dev dk dv dh seq q k v = 
  let qk = matmul q (transpose (Dim 0) (Dim 1) k)
      denom = asTensor'' dev [(sqrt $ fromIntegral dk)::Float]
  in matmul (softmax (Dim 1) $ qk / denom) v
