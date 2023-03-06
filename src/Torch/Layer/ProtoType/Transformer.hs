{-# LANGUAGE DeriveGeneric, RecordWildCards, DisambiguateRecordFields, DuplicateRecordFields #-}

module Torch.Layer.ProtoType.Transformer (
  TransformerHypParams(..)
  , TransformerParams(..)
  , AttentionHypParams(..)
  , AttentionParams(..)
  , sdpAttention
  , positionwiseFeedForward
  , normalize
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
import Torch.Functional   (Dim(..),KeepDim(..),mul,exp,sin,cos,matmul,sqrt,transpose,cat,stack,softmax,sumDim,dropout,unsqueeze,flattenAll,stdMeanDim,chunk)
import Torch.NN           (Parameterized(..),Randomizable(..),Parameter,sample)
--import Torch.Autograd    (IndependentTensor(..),makeIndependent)
--hasktorch-tools
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.MLP    (MLPHypParams(..),MLPParams(..),mlpLayer)
import Torch.Tensor.TensorFactories (asTensor'',zeros',ones')
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
  , nHeads :: Int
  , numLayers :: Int 
  , nonlinearity :: ActName
  } deriving (Eq, Show)

data TransformerParams = TransformerParams {
  inputEmbeddingParams :: LinearParams
  , encoderParamsStack :: [AttentionParams]
  , denomNumerP :: Tensor
} deriving (Generic)
instance Parameterized TransformerParams -- Generic

instance Randomizable TransformerHypParams TransformerParams where
  sample TransformerHypParams{..} = do
    let dimModel = nHeads * dimQK
        denomP = asTensor'' dev $ - (log (10000::Float)) / (fromIntegral dimModel) -- | <> seqLen=5, dimModel=6
        numerP = asTensor'' dev $ [0,2..dimModel-1]                                -- | <dimModel/2>
    TransformerParams
      <$> sample (LinearHypParams dev hasBias dimInput dimModel)
      <*> forM [1..numLayers] (\_ ->
            sample (AttentionHypParams dev hasBias nHeads dimQK dimFF nonlinearity)
            )
      <*> return (denomP * numerP)

data AttentionHypParams = AttentionHypParams {
  dev :: Device
  , hasBias :: Bool
  , nHeads :: Int
  , dimQK :: Int
  , dimFF :: Int
  , nonlinearity :: ActName
} deriving (Eq, Show)

data AttentionParams = AttentionParams {
  genQKV :: LinearParams
  , mlpParams :: MLPParams
  , denomA :: Tensor
} deriving (Generic)
instance Parameterized AttentionParams

instance Randomizable AttentionHypParams AttentionParams where
  sample AttentionHypParams{..} = do
    let dimModel = nHeads * dimQK
    AttentionParams
      <$> sample (LinearHypParams dev hasBias dimModel (dimModel*3))
      <*> sample (MLPHypParams dev dimModel [(dimModel,nonlinearity),(dimFF,nonlinearity),(dimModel,nonlinearity)])
      <*> return (sqrt $ asTensor'' dev [(fromIntegral dimQK)::Float])

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
  let [q,k,v] = input                 -- | <nBatches,seqLen,dimModel>
              .-> normalize dev       -- | <nBatches,seqLen,dimModel>
              .-> linearLayer genQKV  -- | <nBatches,seqLen,dimModel*3>
              .-> chunk 3 (Dim 2)     -- | [<nBatches,seqLen,dimModel>] of length 3
      x = sdpAttention dev nBatches nHeads seqLen dimQK denomA q k v -- | <nBatches,seqLen,dimModel>
          .-> dropoutLayer  -- | <nBatches,seqLen,dimModel>
          .-> (+ input)     -- | <nBatches,seqLen,dimModel>
  return $ x
           .-> mlpLayer mlpParams -- | <nBatches,seqLen,dimModel>
           .-> (+ x)              -- | <nBatches,seqLen,dimModel> -- residual Connection
  where -- | (Tensor -> Tensor) -> Tensor -> Tensor
    dropoutLayer = unsafePerformIO . (dropout dropoutProb True)

-- | scaled dot-product (multi headed) attention
sdpAttention :: Device -- ^ device
  -> Int -- ^ numBatches
  -> Int -- ^ nHeads (of Q,K,V)
  -> Int -- ^ seq length
  -> Int -- ^ dimQK (of Q,K,V)
  -> Tensor -- ^ denominator
  -> Tensor -- ^ Q <nBatches,seqLen,dimModel>
  -> Tensor -- ^ K <nBatches,seqLen,dimModel> 
  -> Tensor -- ^ V <nBatches,seqLen,dimModel>
  -> Tensor -- ^ output <nBatches,seqLen,Heads*dimQK>
sdpAttention dev nBatches nHeads seqLen dimQK denomA q k v = 
  let q' = q                                           -- | <nBatches,seqLen,dimModel>
           .-> reshape [nBatches,seqLen,nHeads,dimQK]  -- | <nBatches,seqLen,nHeads,dimQK>
           .-> transpose (Dim 1) (Dim 2)               -- | <nBatches,nHeads,seqLen,dimQK>
      k' = k                                           -- | <nBatches,seqLen,dimModel>
           .-> reshape [nBatches,seqLen,nHeads,dimQK,1]  -- | <nBatches,seqLen,nHeads,dimQK,1>
           .-> transpose (Dim 1) (Dim 2)               -- | <nBatches,nHeads,seqLen,dimQK,1>
      v' = v                                           -- | <nBatches,seqLen,dimModel>
           .-> reshape [nBatches,seqLen,nHeads,dimQK]   -- | <nBatches,seqLen,nHeads,dimQK>
           .-> transpose (Dim 1) (Dim 2)               -- | <nBatches,nHeads,seqLen,dimQK>
      a = for [0..(seqLen-1)] $ \i -> 
           q'                                       -- | <nBatches,nHeads,seqLen,dimQK>
           .-> sliceDim 2 i (i+1) 1                 -- | <nBatches,nHeads,1,dimQK>
           .-> reshape [nBatches,nHeads,1,1,dimQK]  -- | <nBatches,nHeads,1,1,dimQK>
           .-> (flip matmul) k'                     -- | <nBatches,nHeads,seqLen,1,1> broadcasting q
           .-> reshape [nBatches,nHeads,seqLen,1]   -- | <nBatches,nHeads,seqLen,1>
           .-> softmax (Dim 2)                      -- | <nBatches,nHeads,seqLen,1>
           .-> (\x ->  x / denomA)                   -- | <nBatches,nHeads,seqLen,1>
           .-> (flip mul) v'                        -- | <nBatches,nHeads,seqLen,dimQK>
           .-> sumDim (Dim 2) KeepDim Float         -- | <nBatches,nHeads,1,dimQK>
  in a               -- | [<nBatches,nHeads,1,dimQK>] of length seqLen
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

normalize :: Device -- ^ device
  -> Tensor         -- | <nBatches,seqLen,dimModel>
  -> Tensor         -- | <nBatches,seqLen,dimModel>
normalize dev input = unsafePerformIO $ do
  let inputShape = shape input
      eps = 1e-6
      ones = ones' dev inputShape
      zeros = zeros' dev inputShape
      (std,mean) = stdMeanDim (Dim $ (length inputShape)-1) True KeepDim input
  return $ ((ones * (input - mean)) / (std + eps)) + zeros -- Why add zeros??

positionalEncoding :: Device -- ^ dev
  -> Int -- ^ seqLen
  -> Int    -- ^ dimModel
  -> Tensor -- ^ denom * numer2
  -- -> Tensor -- ^ input tensor <seqLen,dimModel>
  -> Tensor -- ^ output tensor <seqLen,dimModel>
positionalEncoding dev seqLen dimModel denomNumerP = unsafePerformIO $ do
  let position = ((map fromIntegral [0..seqLen-1])::[Float]) -- | [0..seqLen-1]::[Float]
                 .-> asTensor'' dev                        -- | <seqLen>
                 .-> unsqueeze (Dim 1)                     -- | <seqLen,1>
      points = position * exp (denomNumerP) -- | <seqLen,dimModel/2> = <5,3>
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
               .-> (+ positionalEncoding dev seqLen dimModel denomNumerP) -- | <nBatches,seqLen,dimModel>
  -- | foldl' :: (b -> a -> b) -> b -> [a] -> b
  return $ foldl' (\x layer -> layer x)
                  input'
                  (do 
                    attentionParam <- encoderParamsStack
                    return $ attentionLayer attentionParam dev nHeads dimQK dropoutProb)

