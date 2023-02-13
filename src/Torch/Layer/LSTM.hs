{-# LANGUAGE DeriveGeneric #-}
--{-# LANGUAGE DisambiguateRecordFields #-}
{-# LANGUAGE DuplicateRecordFields #-}

module Torch.Layer.LSTM (
  LstmHypParams(..)
  , LstmParams(..)
  , lstmLayers
  , InitialStatesHypParams(..)
  , InitialStatesParams(..)
  , toDependentTensors
  ) where 

import Prelude hiding (tanh) 
import GHC.Generics              --base
import Data.Maybe (isJust)       --base
import Data.List (scanl',foldl',scanr) --base
import Control.Monad (forM,unless)     --base
import System.IO.Unsafe (unsafePerformIO) --base
--hasktorch
import Torch.Tensor       (Tensor(..),shape,select,sliceDim)
import Torch.Functional   (Dim(..),add,sigmoid,cat,stack,dropout)
import Torch.Device       (Device(..))
import Torch.NN           (Parameterized(..),Randomizable(..),Parameter,sample)
import Torch.Autograd (IndependentTensor(..),makeIndependent)
--hasktorch-tools
import Torch.Tensor.Util (unstack)
import Torch.Tensor.TensorFactories (randintIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)

data LstmHypParams = LstmHypParams {
  dev :: Device
  , bidirectional :: Bool 
  , inputSize :: Int  -- ^ The number of expected features in the input x
  , hiddenSize :: Int -- ^ The number of features in the hidden state h
  , numLayers :: Int     -- ^ Number of recurrent layers
  , hasBias :: Bool  -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  , projSize :: Maybe Int -- ^ If > 0, will use LSTM with projections of corresponding size.
  -- , batch_first :: Bool -- ^ If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
  } deriving (Eq, Show)

data SingleLstmParams = SingleLstmParams {
    forgetGate :: LinearParams
    , inputGate :: LinearParams
    , candidateGate :: LinearParams
    , outputGate :: LinearParams
    , projGate :: Maybe LinearParams   -- ^ a model for the projection layer
    } deriving (Show, Generic)
instance Parameterized SingleLstmParams

lstmCell :: SingleLstmParams 
  -> (Tensor,Tensor) -- ^ (ht,ct) 
  -> Tensor          -- ^ xt
  -> (Tensor,Tensor) -- ^ (ht',ct')
lstmCell SingleLstmParams{..} (ht,ct) xt =
  let xt_ht = cat (Dim 0) [xt,ht]
      ft = sigmoid $ linearLayer forgetGate $ xt_ht
      it = sigmoid $ linearLayer inputGate $ xt_ht
      cant = tanh' $ linearLayer candidateGate $ xt_ht
      ct' = (ft * ct) + (it * cant)
      ot = sigmoid $ linearLayer outputGate $ xt_ht
      ht' = ot * (tanh' ct')
  in (ht',ct')
  where -- | HACK : Torch.Functional.tanhとexpが `Segmentation fault`になるため
    tanh' x = 2 * (sigmoid $ 2 * x) - 1

-- | inputのlistから、(cellState,hiddenState=output)のリストを返す
-- | （lstmLayersのサブルーチン。外部から使う想定ではない）
-- | scanl'の型メモ :: ((h,c) -> input -> (h',c')) -> (h0,c0) -> [input] -> [(hi,ci)]
singleLstmLayer :: Bool -- ^ True if BiLSTM, False otherwise
  -> Int                -- ^ stateDim (=hDim)
  -> SingleLstmParams   -- ^ params
  -> (Tensor,Tensor)    -- ^ A pair (h0,c0): shape [1,hDim] for one-directional and [2,hDim] for BiLSTM
  -> Tensor             -- ^ an input tensor of shape [seqLen,iDim] for the 1st-layer and [seqLen,oDim] for the rest
  -> (Tensor,(Tensor,Tensor)) -- ^ an output pair of shape ([seqLen,oDim],([D,oDim],[D,cDim]))
singleLstmLayer bidirectional stateDim singleLstmParams (h0,c0) inputs = unsafePerformIO $ do
  let h0shape = shape h0
      c0shape = shape c0 
      projLayer = case projGate singleLstmParams of
                    Just projG -> (stack (Dim 0)) . map (linearLayer projG) . unstack
--                  Just projG -> linearLayer projP . reshape [] --broadcasting [oDim,projDim] to [seqLen,oDim]
                    Nothing -> id
  if bidirectional -- check the well-formedness of the shapes of h0 and c0
    then do -- the case of BiLSTM
      unless (h0shape == [2,stateDim]) $ ioError $ userError $ "illegal shape of h0 for BiLSTM: " ++ (show h0shape) 
      unless (c0shape == [2,stateDim]) $ ioError $ userError $ "illegal shape of c0 for BiLSTM: " ++ (show c0shape)
      let h0c0f = (select 0 0 h0, select 0 0 c0) -- pick the first (h0,c0) pair for the forward cells
          h0c0b = (select 0 1 h0, select 0 1 c0) -- pick the second (h0,c0) pair for the backward cells
          (hsForward,csForward) = unzip $ tail $ scanl' (lstmCell singleLstmParams) h0c0f $ unstack inputs -- removing (h0,c0) by tail
          (hsBackward,csBackward) = unzip $ init $ scanr (flip $ lstmCell singleLstmParams) h0c0b $ unstack inputs -- removing (h0,c0) by init
          merge = \f b -> stack (Dim 0) $ map (uncurry add) $ zip f b
          hlast = stack (Dim 0) [last hsForward, head hsBackward]
          clast = stack (Dim 0) [last csForward, head csBackward]
      return (projLayer $ merge hsForward hsBackward, (hlast, clast))
    else do -- the case of LSTM
      unless (h0shape == [1,stateDim]) $ ioError $ userError $ "illegal shape of h0 for LSTM: " ++ (show h0shape) 
      unless (c0shape == [1,stateDim]) $ ioError $ userError $ "illegal shape of c0 for LSTM: " ++ (show c0shape)
      let h0c0f = (select 0 0 h0, select 0 0 c0) 
          (hsForward,csForward) = unzip $ tail $ scanl' (lstmCell singleLstmParams) h0c0f $ unstack inputs -- | (c0,h0)は除くためtailを取る
          hlast = last hsForward
          clast = last csForward
      return (projLayer $ stack (Dim 0) hsForward, (hlast, clast))

data LstmParams = LstmParams {
  firstLstmParams :: SingleLstmParams    -- ^ a model for the first LSTM layer
  , restLstmParams :: [SingleLstmParams] -- ^ models for the rest of LSTM layers
  } deriving (Show, Generic)
instance Parameterized LstmParams

instance Randomizable LstmHypParams LstmParams where
  sample LstmHypParams{..} = do
    let xDim = inputSize
        hDim = hiddenSize
        cDim = hiddenSize
        xh1Dim = xDim + hDim
        oDim = case projSize of
                 Just projDim -> projDim
                 Nothing -> hDim
        xh2Dim = oDim + hDim
    LstmParams
      <$> (SingleLstmParams
            <$> sample (LinearHypParams dev hasBias xh1Dim cDim) -- forgetGate
            <*> sample (LinearHypParams dev hasBias xh1Dim cDim) -- inputGate
            <*> sample (LinearHypParams dev hasBias xh1Dim cDim) -- candGate
            <*> sample (LinearHypParams dev hasBias xh1Dim hDim) -- outputGate
            <*> (sequence $ case projSize of 
                  Just projDim -> Just $ sample $ LinearHypParams dev True hDim projDim
                  Nothing -> Nothing)
            )
      <*> forM [2..numLayers] (\_ ->
        SingleLstmParams 
          <$> sample (LinearHypParams dev hasBias xh2Dim cDim)
          <*> sample (LinearHypParams dev hasBias xh2Dim cDim)
          <*> sample (LinearHypParams dev hasBias xh2Dim cDim)
          <*> sample (LinearHypParams dev hasBias xh2Dim hDim)
          <*> (sequence $ case projSize of 
                Just projDim -> Just $ sample $ LinearHypParams dev True hDim projDim
                Nothing -> Nothing)
          )

-- | LSTM layerのメイン関数
lstmLayers :: LstmParams -- ^ parameters (=model)
  -> (Tensor,Tensor) -- ^ a pair of initial tensors: [D*numLayers,hiddenSize]
  -> Maybe Double    -- ^ introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.
  -> Tensor          -- ^ an input tensor of shape [seqLen,inputSize]
  -> (Tensor,(Tensor,Tensor)) -- ^ an output of shape ([seqLen,projSize],([seqLen,D*hiddenSize],[seqLen,D*hiddenSize]) 
lstmLayers LstmParams{..} (h0,c0) dropoutProb inputs = unsafePerformIO $ do
  let numLayers = length restLstmParams + 1
      (dnumLayers:(hiddenSize:_)) = shape h0
      bidirectional | dnumLayers == numLayers * 2 = True
                    | dnumLayers == numLayers = False
                    | otherwise = False -- ここはエラーメッセージを出すべきかどうか
      (h0h:h0t) = if bidirectional -- check the input shapes of h0 tensor
                    then [sliceDim 0 (2*i) (2*i+2) 1 h0 | i <- [0..numLayers]]
                    else [sliceDim 0 i (i+1) 1 h0 | i <- [0..numLayers]]
      (c0h:c0t) = if bidirectional -- check the input shapes of c0 tensor
                    then [sliceDim 0 (2*i) (2*i+2) 1 c0 | i <- [0..numLayers]]
                    else [sliceDim 0 i (i+1) 1 c0 | i <- [0..numLayers]]
      firstLayer = singleLstmLayer bidirectional hiddenSize firstLstmParams (h0h,c0h) 
      restOfLayers = map (uncurry $ singleLstmLayer bidirectional hiddenSize) $ zip restLstmParams $ zip h0t c0t
      dropoutLayer = case dropoutProb of
                       Just prob -> unsafePerformIO . (dropout prob True)
                       Nothing -> id
      stackedLayers = \inputTensor -> 
                        scanl'
                          (\ohc nextLayer -> (nextLayer . dropoutLayer . fst) ohc)
                          (firstLayer inputTensor)
                          restOfLayers
      (output, hncn) = unzip $ stackedLayers inputs
      (hn, cn) = unzip hncn
  return (last output, (stack (Dim 0) hn, stack (Dim 0) cn))

data InitialStatesHypParams = InitialStatesHypParams {
  dev :: Device
  , bidirectional :: Bool
  , hiddenSize :: Int
  , numLayers :: Int
  } deriving (Eq, Show)

data InitialStatesParams = InitialStatesParams {
  h0 :: Parameter 
  , c0 :: Parameter
  } deriving (Show, Generic)
instance Parameterized InitialStatesParams

instance Randomizable InitialStatesHypParams InitialStatesParams where
  sample InitialStatesHypParams{..} = 
    InitialStatesParams
      <$> (makeIndependent =<< randintIO' dev (-1) 1 [(if bidirectional then 2 else 1) * numLayers, hiddenSize])
      <*> (makeIndependent =<< randintIO' dev (-1) 1 [(if bidirectional then 2 else 1) * numLayers, hiddenSize])

toDependentTensors :: InitialStatesParams -> (Tensor,Tensor)
toDependentTensors InitialStatesParams{..} = (toDependent h0,toDependent c0)