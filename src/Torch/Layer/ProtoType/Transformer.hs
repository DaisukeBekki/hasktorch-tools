{-# LANGUAGE DeriveGeneric, RecordWildCards, DisambiguateRecordFields, DuplicateRecordFields #-}

module Torch.Layer.ProtoType.Transformer (
  TransformerHypParams(..)
  , TransformerParams(..)
  , AttentionHypParams(..)
  , AttentionParams(..)
  , sdpAttention
  , positionwiseFeedForward
  , attentionLayer
  , positionalEncoding
  , encoder
  ) where 

import Prelude hiding (sqrt,sin,cos,exp)
import GHC.Generics                         --base
import Data.List          (foldl')          --base
import Control.Monad      (forM,unless)     --base
import System.IO.Unsafe   (unsafePerformIO) --base
import Data.Function      ((&))             --base
--hasktorch
import Torch.Tensor       (Tensor(..),shape,reshape,sliceDim)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.DType        (DType(..))
import Torch.Functional   (Dim(..),KeepDim(..),mul,exp,sin,cos,matmul,sqrt,transpose,cat,stack,softmax,sumDim,dropout,unsqueeze,flattenAll)
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

debug :: Bool
debug = True

data TransformerHypParams = TransformerHypParams {
  dev :: Device
  , hasBias :: Bool  -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  , dimInput :: Int  -- ^ The number of expected features in the input x
  , dimQK :: Int 
  , dimFF :: Int
  , numHeads :: Int
  , numLayers :: Int 
  , nonlinearity :: ActName
  } deriving (Eq, Show)

data TransformerParams = TransformerParams {
  inputEmbeddingParams :: LinearParams
  , encoderParamsStack :: [AttentionParams]
} deriving (Generic)
instance Parameterized TransformerParams -- Generic

instance Randomizable TransformerHypParams TransformerParams where
  sample TransformerHypParams{..} = do
    let dimModel = numHeads * dimQK
    TransformerParams
      <$> sample (LinearHypParams dev hasBias dimInput dimModel)
      <*> forM [1..numLayers] (\_ ->
            sample (AttentionHypParams dev hasBias dimModel dimFF nonlinearity)
            )

data AttentionHypParams = AttentionHypParams {
  dev :: Device
  , hasBias :: Bool
  , dimModel :: Int
  , dimFF :: Int
  , nonlinearity :: ActName
} deriving (Eq, Show)

data AttentionParams = AttentionParams {
  genQ :: LinearParams
  , genK :: LinearParams
  , genV :: LinearParams
  , mlpParams :: MLPParams
} deriving (Generic)
instance Parameterized AttentionParams

instance Randomizable AttentionHypParams AttentionParams where
  sample AttentionHypParams{..} = do
    AttentionParams
      <$> sample (LinearHypParams dev hasBias dimModel dimModel)
      <*> sample (LinearHypParams dev hasBias dimModel dimModel)
      <*> sample (LinearHypParams dev hasBias dimModel dimModel)
      <*> sample (MLPHypParams dev dimModel [(dimModel,nonlinearity),(dimFF,nonlinearity),(dimModel,nonlinearity)])

attentionLayer :: AttentionParams 
  -> Device 
  -> Int    -- ^ nHeads
  -> Int    -- ^ dimQK
  -> Double -- ^ dropoutProb
  -> Tensor -- ^ input tensor <nBatches,seqLen,dimModel>
  -> Tensor -- ^ output tensor <nBatches,seqLen,dimModel>
attentionLayer AttentionParams{..} dev nHeads dimQK dropoutProb input = unsafePerformIO $ do
  let [nBatches,seqLen,dimModel] = shape input
  unless (dimModel == nHeads * dimQK) $ ioError $ userError $ "illegal input shape: " ++ (show $ shape input)
  let q = linearLayer genQ input -- | <nBatches,seqLen,dimModel>
      k = linearLayer genK input -- | <nBatches,seqLen,dimModel>
      v = linearLayer genV input -- | <nBatches,seqLen,dimModel>
      x = sdpAttention dev nBatches nHeads seqLen dimQK q k v -- | <nBatches,seqLen,dimModel>
          .-> normalize    -- | <nBatches,seqLen,dimModel>
          .-> dropoutLayer -- | <nBatches,seqLen,dimModel>
          .-> (+ input)    -- | <nBatches,seqLen,dimModel>
  return $ x
           .-> mlpLayer mlpParams -- | <nBatches,seqLen,dimModel>
           .-> normalize          -- | <nBatches,seqLen,dimModel>
           .-> (+ x)              -- | <nBatches,seqLen,dimModel>
  where -- | (Tensor -> Tensor) -> Tensor -> Tensor
    --residualConnection sublayer x = x +  $ sublayer $ normalize x)
    dropoutLayer = unsafePerformIO . (dropout dropoutProb True)

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
      v' = v                                           -- | <nBatches,seqLen,dimModel>
           .-> reshape [nBatches,seqLen,nHeads,dimQK]   -- | <nBatches,seqLen,nHeads,dimQK>
           .-> transpose (Dim 1) (Dim 2)               -- | <nBatches,nHeads,seqLen,dimQK>
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
           .-> reshape [nBatches,seqLen,nHeads*dimQK]-- | <nBatches,seqLen,nHeads*dimQK=dimModel>

positionwiseFeedForward :: MLPParams -- ^ 
  -> Device -- ^ device
  -> Int    -- ^ dimFF (of the hidden layer of MLP)
  -> Tensor -- ^ input <nBatches,seqLen,dimModel>
  -> Tensor -- ^ output <nBatches,seqLen,dimModel>
positionwiseFeedForward mlpParams dev dimFF input = 
  mlpLayer mlpParams $ input

normalize :: Tensor -> Tensor
normalize = id

positionalEncoding :: Device -- ^ dev
  -> Int -- ^ seqLen
  -> Int    -- ^ dimModel
  -- -> Tensor -- ^ input tensor <seqLen,dimModel>
  -> Tensor -- ^ output tensor <seqLen,dimModel>
positionalEncoding dev seqLen dimModel = unsafePerformIO $ do
  let position = ((map fromIntegral [0..seqLen-1])::[Float]) -- | [0..seqLen-1]::[Float]
                 .-> asTensor'' dev                        -- | <seqLen>
                 .-> unsqueeze (Dim 1)                     -- | <seqLen,1>
      denom = asTensor'' dev $ - (log (10000::Float)) / (fromIntegral dimModel) -- | <> seqLen=5, dimModel=6
      numer = asTensor'' dev $ [0,2..dimModel-1]                                  -- | <dimModel/2>
      points = position * exp (denom * numer) -- | <seqLen,dimModel/2> = <5,3>
      even = unsqueeze (Dim 1) $ flattenAll $ sin points -- | <seqLen * dimModel/2>
      odd  = unsqueeze (Dim 1) $ flattenAll $ cos points -- | <seqLen * dimModel/2>
  return $ [even,odd]       -- | [<seqLen * dimModel/2>] of length 2
           .-> cat (Dim 1)  -- | <seqLen*dimModel/2, 2>
           .-> reshape [seqLen,dimModel] -- | <seqLen,dimModel>

encoder :: TransformerParams
  -> Device -- ^ dev
  -> Int    -- ^ nHeads
  -> Int    -- ^ dimQK
  -> Double -- ^ dropoutProb
  -> Tensor -- ^ input tensor  <nBatches,seqLen,dimInput>
  -> Tensor -- ^ output tensor <nBatches,seqLen,dimModel
encoder TransformerParams{..} dev nHeads dimQK dropoutProb input = unsafePerformIO $ do
  let [nBatches,seqLen,dimInput] = shape input
      dimModel = nHeads * dimQK
      y = positionalEncoding dev seqLen dimModel
      input' = input                                          -- | <nBatches,seqLen,dimInput>
               .-> linearLayer inputEmbeddingParams           -- | <nBatches,seqLen,dimModel>
               .-> (+ positionalEncoding dev seqLen dimModel) -- | <nBatches,seqLen,dimModel>
  -- | foldl' :: (b -> a -> b) -> b -> [a] -> b
  return $ foldl' (\x layer -> layer x)
                  input'
                  (do 
                    attentionParam <- encoderParamsStack
                    return $ attentionLayer attentionParam dev nHeads dimQK dropoutProb)

