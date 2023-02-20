{-# LANGUAGE DeriveGeneric, RecordWildCards, DisambiguateRecordFields, DuplicateRecordFields #-}

module Torch.Layer.ProtoType.Transformer (
  TransformerHypParams(..)
  , TransformerParams(..)
  , sdpAttention
  ) where 

import GHC.Generics                         --base
import System.IO.Unsafe   (unsafePerformIO) --base
import Data.Functor       ((<&>))           --base
--hasktorch
import Torch.Tensor       (Tensor(..),shape,reshape)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.Functional   (Dim(..),matmul,transpose,softmax)
import Torch.NN           (Parameterized(..),Randomizable(..),Parameter,sample)
--import Torch.Autograd    (IndependentTensor(..),makeIndependent)
--hasktorch-tools
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Tensor.TensorFactories (asTensor'',randnIO')

data TransformerHypParams = TransformerHypParams {
  dev :: Device
  , hasBias :: Bool  -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  , dimInput :: Int  -- ^ The number of expected features in the input x
  , dimQK :: Int 
  , dimV :: Int
  , dimModel :: Int
  , numHeads :: Int
  , numLayers :: Int 
  } deriving (Eq, Show)

data TransformerParams = TransformerParams {
  embed :: LinearParams
  , qGen :: LinearParams
  , kGen :: LinearParams
} deriving (Generic)
instance Parameterized TransformerParams -- Generic

instance Randomizable TransformerHypParams TransformerParams where
  sample TransformerHypParams{..} = 
    TransformerParams
      <$> sample (LinearHypParams dev hasBias dimInput dimModel)
      <*> sample (LinearHypParams dev hasBias dimModel (numHeads * dimQK))
      <*> sample (LinearHypParams dev hasBias dimModel (numHeads * dimQK))

-- | Backwards function composition. This is a specialization of '<&>', but it
-- has a different fixity.
(-.) :: (a -> b) -> (b -> c) -> a -> c
(-.) = (<&>)

-- | scaled dot-product attention
sdpAttention :: TransformerParams -- ^ model
  -> Int -- ^ dimQK (of Q,K)
  -> Int -- ^ dimV (of V)
  -> Int -- ^ dimModel
  -> Int -- ^ numBatches
  -> Int -- ^ numHeads (of Q,K,V)
  -> Int -- ^ seq length
  -> Tensor -- ^ Q <nBatches,seqLen,dimModel>
  -> Tensor -- ^ K <nBatches,seqLen,dimModel> 
  -> Tensor -- ^ V <nBatches,seqLen,vDim>
  -> Tensor -- ^ output <nBatches,seqLen,dimModel>
sdpAttention TransformerParams{..} dimQK dimV dimModel nBatches nHeads seqLen q k v = unsafePerformIO $ do
  -- | q,k:                                       <nBatches,seqLen,dimModel>
  let qH = linearLayer qGen                           -- | -> <nBatches,seqLen,nHeads*dimQK>
           <&> reshape [nBatches,seqLen,nHeads,dimQK] -- | -> <nBatches,seqLen,nHeads,dimQK>
           <&> transpose (Dim 1) (Dim 2)              -- | -> <nBatches,nHeads,seqLen,dimQK>
      --transpose (Dim 1) (Dim 2) $ reshape [nBatches,seqLen,nHeads,dimQK] $ linearLayer qGen q
      --kH = transpose (Dim 1) (Dim 2) $ reshape [nBatches,seqLen,nHeads,dimQK] $ linearLayer kGen k
  --  qk = matmul q (transpose (Dim 0) (Dim 1) k)
  --  denom = asTensor'' dev [(sqrt $ fromIntegral kDim)::Float]
  --  matmul (softmax (Dim 1) $ qk / denom) v
  return $ qH k

