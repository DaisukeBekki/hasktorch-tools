{-# LANGUAGE DeriveGeneric, RecordWildCards, DisambiguateRecordFields, DuplicateRecordFields #-}

module Torch.Layer.ProtoType.Transformer (
  TransformerHypParams(..)
  , TransformerParams(..)
  , sdpAttention
  , positionwiseFeedForward
  , positionalEncoding
  ) where 

import Prelude hiding (sqrt,sin,cos,exp)
import GHC.Generics                         --base
import Control.Monad      (forM)            --base
import System.IO.Unsafe   (unsafePerformIO) --base
import Data.Function      ((&))             --base
--hasktorch
import Torch.Tensor       (Tensor(..),shape,reshape,sliceDim)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.DType        (DType(..))
import Torch.Functional   (Dim(..),KeepDim(..),mul,exp,sin,cos,matmul,sqrt,transpose,cat,stack,softmax,sumDim,dropout)
import Torch.NN           (Parameterized(..),Randomizable(..),Parameter,sample)
--import Torch.Autograd    (IndependentTensor(..),makeIndependent)
--hasktorch-tools
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.MLP    (MLPHypParams(..),MLPParams(..),mlpLayer)
import Torch.Tensor.TensorFactories (asTensor'',zeros')
import Torch.Layer.NonLinear (ActName(..))

-- | Backwards function application.
(.->) :: a -> (a -> b) -> b
(.->) = (&)

for :: [a] -> (a -> b) -> [b]
for = flip fmap

data TransformerHypParams = TransformerHypParams {
  dev :: Device
  , hasBias :: Bool  -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  , dimInput :: Int  -- ^ The number of expected features in the input x
  , dimQK :: Int 
  , numHeads :: Int
  , numLayers :: Int 
  , dropoutProb :: Double
  , mlpSpec :: [(Int,ActName)]
  } deriving (Eq, Show)

data AttentionParams = AttentionParams {
  genQ :: LinearParams
  , genK :: LinearParams
  , genV :: LinearParams
  , mlpParams :: MLPParams
} deriving (Generic)
instance Parameterized AttentionParams

data TransformerParams = TransformerParams {
  inputEmbedding :: LinearParams
  , encoderStack :: [AttentionParams]
} deriving (Generic)
instance Parameterized TransformerParams -- Generic

instance Randomizable TransformerHypParams TransformerParams where
  sample TransformerHypParams{..} = do
    let dimModel = numHeads * dimQK
    TransformerParams
      <$> sample (LinearHypParams dev hasBias dimInput dimModel)
      <*> forM [1..numLayers] (\_ ->
            AttentionParams 
              <$> sample (LinearHypParams dev hasBias dimModel dimModel) -- genQ
              <*> sample (LinearHypParams dev hasBias dimModel dimModel) -- genK
              <*> sample (LinearHypParams dev hasBias dimModel dimModel) -- genV
              <*> sample (MLPHypParams dev dimModel mlpSpec)             -- MLP
      )

-- | scaled dot-product attention
sdpAttention :: Device -- ^ device
  -> Int -- ^ numBatches
  -> Int -- ^ numHeads (of Q,K,V)
  -> Int -- ^ seq length
  -> Int -- ^ dimQK (of Q,K,V)
  -> Tensor -- ^ Q <nBatches,seqLen,dimModel>
  -> Tensor -- ^ K <nBatches,seqLen,dimModel> 
  -> Tensor -- ^ V <nBatches,seqLen,dimModel>
  -> Tensor -- ^ output <nBatches,seqLen,Heads*dimQK>
sdpAttention dev nBatches nHeads seqLen dimQK q k v = unsafePerformIO $ do
  let q' = q                                           -- | <nBatches,seqLen,dimModel>
           .-> reshape [nBatches,seqLen,nHeads,dimQK]  -- | <nBatches,seqLen,nHeads,dimQK>
           .-> transpose (Dim 1) (Dim 2)               -- | <nBatches,nHeads,seqLen,dimQK>
      k' = k                                           -- | <nBatches,seqLen,dimModel>
           .-> reshape [nBatches,seqLen,nHeads,dimQK,1]  -- | <nBatches,seqLen,nHeads,dimQK,1>
           .-> transpose (Dim 1) (Dim 2)               -- | <nBatches,nHeads,seqLen,dimQK,1>
      v' = v                                           -- | <nBatches,nHeads,seqLen,dimModel>
           .-> reshape [nBatches,seqLen,nHeads,dimQK]   -- | <nBatches,seqLen,nHeads,dimV>
           .-> transpose (Dim 1) (Dim 2)               -- | <nBatches,nHeads,seqLen,dimV>
      denom = sqrt $ asTensor'' dev [(fromIntegral dimQK)::Float] -- | denominator
      a = for [0..(seqLen-1)] $ \i -> 
           q'                                       -- | <nBatches,nHeads,seqLen,dimQK>
           .-> sliceDim 2 i (i+1) 1                 -- | <nBatches,nHeads,1,dimQK>
           .-> reshape [nBatches,nHeads,1,1,dimQK]  -- | <nBatches,nHeads,1,1,dimQK>
           .-> (flip matmul) k'                     -- | <nBatches,nHeads,seqLen,1,1> broadcasting q
           .-> reshape [nBatches,nHeads,seqLen,1]   -- | <nBatches,nHeads,seqLen,1>
           .-> softmax (Dim 2)                      -- | <nBatches,nHeads,seqLen,1>
           .-> (\x ->  x / denom)                   -- | <nBatches,nHeads,seqLen,1>
           .-> (flip mul) v'                        -- | <nBatches,nHeads,seqLen,dimQK>
           .-> sumDim (Dim 2) KeepDim Float         -- | <nBatches,nHeads,1,dimQK>
  return $ a               -- | [<nBatches,nHeads,1,dimQK>] of length seqLen
           .-> cat (Dim 2) -- | <nBatches,nHeads,seqLen,dimQK>
           .-> transpose (Dim 1) (Dim 2) -- | <nBatches,seqLen,nHeads,dimQK>
           .-> reshape [nBatches,seqLen,nHeads*dimQK]-- | <nBatches,seqLen,nHeads*dimQK>

positionwiseFeedForward :: MLPParams -- ^ 
  -> Device -- ^ device
  -> Int    -- ^ dimFF (of the hidden layer of MLP)
  -> Tensor -- ^ input <nBatches,seqLen,dimModel>
  -> Tensor -- ^ output <nBatches,seqLen,dimModel>
positionwiseFeedForward mlpParams dev dimFF input = 
  mlpLayer mlpParams $ input

residualConnection :: (Tensor -> Tensor) -> Tensor -> Tensor
residualConnection sublayer x = x + (dropoutLayer (0.1) $ sublayer $ normalize x)

dropoutLayer :: Double -> Tensor -> Tensor
dropoutLayer prob = unsafePerformIO . (dropout prob True)

normalize :: Tensor -> Tensor
normalize = id

positionalEncoding :: Device -- ^ dev
  -> Int -- ^ seqLen
  -> Int    -- ^ dimModel
  -- -> Tensor -- ^ input tensor <seqLen,dModel>
  -> Tensor -- ^ output tensor <seqLen,dModel>
positionalEncoding dev seqLen dimModel = unsafePerformIO $ do
  let position = ((map fromIntegral [0..seqLen])::[Float]) -- | [0..seqLen]::[Float]
                 .-> asTensor'' dev                        -- | <seqLen+1>
                 .-> reshape [seqLen+1,1]                  -- | <seqLen+1,1>
      denom = asTensor'' dev $ - (log (10000::Float)) / (fromIntegral dimModel)           -- Float
      numer = asTensor'' dev $ [0,2..dimModel]
      points = position * exp (denom * numer)
      evenElems = sin points
      oddElems = cos points
  return $ oddElems