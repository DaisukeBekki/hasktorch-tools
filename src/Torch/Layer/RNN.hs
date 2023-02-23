{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}

{- Throuout the comments, a tensor of shape [a,b,..] is written as <a,b,...> -}

module Torch.Layer.RNN (
  RnnHypParams(..)
  , RnnParams(..)
  , singleRnnLayer
  , rnnLayers
  , InitialStatesHypParams(..)
  , InitialStatesParams(..)
  ) where 

import Prelude hiding   (tanh) 
import GHC.Generics              --base
import Data.Function    ((&))    --base
import Data.Maybe       (isJust) --base
import Data.List        (scanl',foldl',scanr) --base
import Control.Monad    (forM,unless)     --base
import System.IO.Unsafe (unsafePerformIO) --base
--hasktorch
import Torch.Tensor      (Tensor(..),shape,select,sliceDim,reshape)
import Torch.Functional  (Dim(..),add,sigmoid,cat,stack,dropout,transpose)
import Torch.Device      (Device(..))
import Torch.NN          (Parameterized(..),Randomizable(..),Parameter,sample)
import Torch.Autograd    (makeIndependent)
--hasktorch-tools
import Torch.Tensor.Util (unstack)
import Torch.Tensor.TensorFactories (randintIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.NonLinear (ActName(..),decodeAct)

(.->) :: a -> (a -> b) -> b
(.->) = (&)

data RnnHypParams = RnnHypParams {
  dev :: Device
  , bidirectional :: Bool -- ^ True if BiLSTM, False otherwise
  , inputSize :: Int  -- ^ The number of expected features in the input x
  , hiddenSize :: Int -- ^ The number of features in the hidden state h
  , numLayers :: Int  -- ^ Number of recurrent layers
  , hasBias :: Bool   -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  -- , batch_first :: Bool -- ^ If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
  } deriving (Eq, Show)

newtype SingleRnnParams = SingleRnnParams {
    rnnGate :: LinearParams
    } deriving (Show, Generic)
instance Parameterized SingleRnnParams

rnnCell :: SingleRnnParams 
  -> Tensor -- ^ ht of shape <hDim>
  -> Tensor -- ^ xt of shape <iDim/oDim>
  -> Tensor -- ^ ht' of shape <hDim>
rnnCell SingleRnnParams{..} ht xt = linearLayer rnnGate $ cat (Dim 0) [xt,ht]

-- | inputのlistから、(cellState,hiddenState=output)のリストを返す
-- | （rnLayersのサブルーチン。外部から使う想定ではない）
-- | scanl'の型メモ :: ((h,c) -> input -> (h',c')) -> (h0,c0) -> [input] -> [(hi,ci)]
singleRnnLayer :: Bool -- ^ bidirectional (True if bidirectioal, False otherwise)
  -> Int               -- ^ stateDim (=hDim)
  -> ActName           -- ^ actname (the name of nonlinear function)
  -> SingleRnnParams   -- ^ singleRnnParams
  -> Tensor            -- ^ h0: <1,hDim> for one-directional and <2,hDim> for BiLSTM
  -> Tensor            -- ^ inputs: <seqLen,iDim/oDim> for the 1st-layer/the rest
  -> (Tensor,Tensor)   -- ^ an output pair (<seqLen,D*oDim>,<D*oDim>)
singleRnnLayer bidirectional stateDim actname singleRnnParams h0 inputs = unsafePerformIO $ do
  let h0shape = shape h0
      [seqLen,_] = shape inputs
      d = if bidirectional then 2 else 1
      actf = decodeAct actname
  unless (h0shape == [d,stateDim]) $ ioError $ userError $ "illegal shape of h0: " ++ (show h0shape) 
  if bidirectional -- check the well-formedness of the shapes of h0 and c0
    then do -- the case of BiRNN
      let h0f = select 0 0 h0 -- | pick the first h0 for the forward cells
          h0b = select 0 1 h0 -- | pick the second h0 for the backward cells
          hsForward = inputs  -- | <seqLen,iDim/oDim> 
            .-> unstack       -- | [<iDim/oDim>] of length seqLen
            .-> scanl' (rnnCell singleRnnParams) h0f -- | [<hDim>] of length seqLen+1
            .-> tail          -- | [<hDim>] of length seqLen (by removing h0f)
            .-> stack (Dim 0) -- | <seqLen, hDim>
          hsBackward = inputs -- | <seqLen,iDim/oDim> 
            .-> unstack       -- | [<iDim/oDim>] of length seqLen
            .-> scanr (flip $ rnnCell singleRnnParams) h0b -- | [<hDim>] of length seqLen+1
            .-> init          -- | [<hDim>] of length seqLen (by removing h0b)
            .-> stack (Dim 0) -- | <seqLen, hDim>
          output = [hsForward, hsBackward]   -- | [<seqLen, hDim>] of length 2
            .-> stack (Dim 0)                -- | <2, seqLen, hDim>
            .-> actf                         -- | <2, seqLen, hDim> ??
            .-> transpose (Dim 0) (Dim 1)    -- | <seqLen, 2, hDim>
            .-> reshape [seqLen, 2*stateDim] -- | <seqLen, 2*hDim>
          hLast = output                           -- | <seqLen, 2*hDim>
            .-> sliceDim 0 (seqLen-1) seqLen 1     -- | <1, 2*hDim>
            .-> (\o -> reshape (tail $ shape o) o) -- | <2*hDim> 
      return (output, hLast)
    else do -- the case of RNN
      let h0f = select 0 0 h0
          output = inputs
            .-> unstack          -- | [<iDim/oDim>] of length seqLen
            .-> scanl' (rnnCell singleRnnParams) h0f -- | [<hDim>] of length seqLen+1
            .-> tail             -- | [<hDim>] of length seqLen (by removing h0)
            .-> stack (Dim 0)    -- | <seqLen, hDim>
            .-> actf             -- | <seqLen, hDim> ??
          hLast = output                           -- | <seqLen, hDim>
            .-> sliceDim 0 (seqLen-1) seqLen 1     -- | <1, hDim>
            .-> (\o -> reshape (tail $ shape o) o) -- | <hDim>
      return (output, hLast)

data RnnParams = RnnParams {
  firstRnnParams :: SingleRnnParams    -- ^ a model for the first RNN layer
  , restRnnParams :: [SingleRnnParams] -- ^ models for the rest of RNN layers
  } deriving (Show, Generic)
instance Parameterized RnnParams

instance Randomizable RnnHypParams RnnParams where
  sample RnnHypParams{..} = do
    let xDim = inputSize
        hDim = hiddenSize
        xh1Dim = xDim + hDim
        d = if bidirectional then 2 else 1
        xh2Dim = (d * hDim) + hDim
    RnnParams
      <$> (SingleRnnParams <$> sample (LinearHypParams dev hasBias xh1Dim hDim)) -- gate
      <*> forM [2..numLayers] (\_ ->
        SingleRnnParams <$> sample (LinearHypParams dev hasBias xh2Dim hDim)
        )

-- | The main function for RNN layers
rnnLayers :: RnnParams -- ^ parameters (=model)
  -> ActName         -- ^ name of nonlinear function
  -> Maybe Double    -- ^ introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.
  -> Tensor          -- ^ an initial tensor: <D*numLayers,hDim>
  -> Tensor          -- ^ an input tensor <seqLen,iDim>
  -> (Tensor,Tensor) -- ^ an output of (<seqLen,D*oDim>,(<D*numLayers,oDim>,<D*numLayers,cDim>))
rnnLayers RnnParams{..} actname dropoutProb h0 inputs = unsafePerformIO $ do
  let numLayers = length restRnnParams + 1
      (dnumLayers:(hiddenSize:_)) = shape h0
  unless (dnumLayers == numLayers * 2 || dnumLayers == numLayers) $ 
    ioError $ userError $ "illegal shape of h0: dnumLayers = " ++ (show dnumLayers) ++ "\nnumLayers = " ++ (show numLayers) 
  let bidirectional | dnumLayers == numLayers * 2 = True
                    | dnumLayers == numLayers = False
                    | otherwise = False -- Unexpected
      d = if bidirectional then 2 else 1
      (h0h:h0t) = [sliceDim 0 (d*i) (d*(i+1)) 1 h0 | i <- [0..numLayers]]
      firstLayer = singleRnnLayer bidirectional hiddenSize actname firstRnnParams h0h
      restOfLayers = map (uncurry $ singleRnnLayer bidirectional hiddenSize actname) $ zip restRnnParams h0t 
      dropoutLayer = case dropoutProb of
                       Just prob -> unsafePerformIO . (dropout prob True)
                       Nothing -> id
      stackedLayers = \inputTensor -> 
                        scanr
                          (\nextLayer ohc -> nextLayer $ dropoutLayer $ fst ohc)
                          (firstLayer inputTensor) -- (<seqLen,D*oDim>,(<D*oDim>,<D*cDim>))
                          restOfLayers -- 
      (outputList, hn) = inputs -- | <seqLen,iDim>
        .-> stackedLayers       -- | [(<seqLen, D*oDim>,<D*oDim>)] of length numLayers
        .-> unzip               -- | ([<seqLen, D*oDim>] of length numLayers, [(<D*oDim>,<D*cDim>)] of length numLayers)
      output = head outputList  -- | [<seqLen, D*oDim>] of length numLayers
      rh = hn                                      -- | 
           .-> stack (Dim 0)                       -- | <numLayers, D*oDim/cDim>
           .-> reshape [d * numLayers, hiddenSize] -- | <D*numLayers, oDim/cDim>
  return (output, rh)

data InitialStatesHypParams = InitialStatesHypParams {
  dev :: Device
  , bidirectional :: Bool
  , hiddenSize :: Int
  , numLayers :: Int
  } deriving (Eq, Show)

newtype InitialStatesParams = InitialStatesParams {
  h0s :: Parameter 
  } deriving (Show, Generic)
instance Parameterized InitialStatesParams

instance Randomizable InitialStatesHypParams InitialStatesParams where
  sample InitialStatesHypParams{..} = 
    InitialStatesParams
      <$> (makeIndependent =<< randintIO' dev (-1) 1 [(if bidirectional then 2 else 1) * numLayers, hiddenSize])

