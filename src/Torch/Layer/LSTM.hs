{-# LANGUAGE DeriveGeneric #-}
--{-# LANGUAGE DisambiguateRecordFields #-}
{-# LANGUAGE DuplicateRecordFields #-}

{- Throuout the comments, a tensor of shape [a,b,..] is written as <a,b,...> -}

module Torch.Layer.LSTM (
  LstmHypParams(..)
  , LstmParams(..)
  , singleLstmLayer
  , lstmLayers
  , InitialStatesHypParams(..)
  , InitialStatesParams(..)
  , toDependentTensors
  ) where 

import Prelude hiding   (tanh) 
import GHC.Generics              --base
import Data.Function    ((&))    --base
import Data.List        (scanl',scanr,singleton) --base
import Control.Monad    (forM,unless)     --base
import System.IO.Unsafe (unsafePerformIO) --base
--hasktorch
import Torch.Tensor      (Tensor(..),shape,select,sliceDim,reshape)
import Torch.Functional  (Dim(..),sigmoid,cat,stack,dropout)
import Torch.Device      (Device(..))
import Torch.NN          (Parameterized(..),Randomizable(..),Parameter,sample)
import Torch.Autograd    (IndependentTensor(..),makeIndependent)
--hasktorch-tools
import Torch.Tensor.Util (unstack)
import Torch.Tensor.TensorFactories (randintIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)

(.->) :: a -> (a -> b) -> b
(.->) = (&)

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
  -> (Tensor,Tensor) -- ^ (ht,ct) of shape (<hDim>,<cDim>)
  -> Tensor          -- ^ xt of shape <iDim/oDim>
  -> (Tensor,Tensor) -- ^ (ht',ct') of shape (<hDim>,<cDim>)
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
  -> Int                -- ^ stateDim (=hDim=cDim)
  -> SingleLstmParams   -- ^ params
  -> (Tensor,Tensor)    -- ^ A pair (h0,c0): <1,hDim> for one-directional and <2,hDim> for BiLSTM
  -> Tensor             -- ^ an input tensor <seqLen,iDim> for the 1st-layer and <seqLen,oDim> for the rest
  -> (Tensor,(Tensor,Tensor)) -- ^ an output pair (<seqLen,D*oDim>,(<D*oDim>,<D*cDim>))
singleLstmLayer bidirectional stateDim singleLstmParams (h0,c0) inputs = unsafePerformIO $ do
  let h0shape = shape h0
      c0shape = shape c0 
      [seqLen,_] = shape inputs
      d = if bidirectional then 2 else 1
      projLayer = case projGate singleLstmParams of
                    Just projParam -> linearLayer projParam -- | <d, seqLen, projDim>
                    Nothing -> id                           -- | <d, seqLen, hDim>
  unless (h0shape == [d,stateDim]) $ ioError $ userError $ "illegal shape of h0: " ++ (show h0shape) 
  unless (c0shape == [d,stateDim]) $ ioError $ userError $ "illegal shape of c0: " ++ (show c0shape)
  if bidirectional -- check the well-formedness of the shapes of h0 and c0
    then do -- the case of BiLSTM
      let h0c0f = (select 0 0 h0, select 0 0 c0) -- | pick the first (h0,c0) pair for the forward cells
          h0c0b = (select 0 1 h0, select 0 1 c0) -- | pick the second (h0,c0) pair for the backward cells
          (hsForward,csForward) = inputs -- | <seqLen,iDim/oDim> 
            .-> unstack  -- | [<iDim/oDim>] of length seqLen
            .-> scanl' (lstmCell singleLstmParams) h0c0f -- | [(<hDim>, <cDim>)] of length seqLen+1
            .-> tail     -- | [(<hDim>, <cDim>)] of length seqLen (by removing (h0,c0))
            .-> unzip    -- | ([<hDim>], [<cDim>])
          (hsBackward,csBackward) = inputs 
            .-> unstack  -- | [<iDim/oDim>] of length seqLen
            .-> scanr (flip $ lstmCell singleLstmParams) h0c0b -- | [(<hDim>, <cDim>)] of length seqLen+1
            .-> init     -- | [(<hDim>, <cDim>)] of length seqLen (by removing (h0,c0))
            .-> unzip    -- | ([<hDim>], [<cDim>])
          cLast = [last csForward, head csBackward] -- | [<cDim>] of length 2
            .-> cat (Dim 0)      -- | <2*cDim>
          output = [stack (Dim 0) hsForward, stack (Dim 0) hsBackward] -- | [<seqLen, hDim>] of length 2
            .-> stack (Dim 0)    -- | <2, seqLen, hDim>
            .-> projLayer        -- | <2, seqLen, projDim/hDim>
            .-> unstack          -- | [<seqLen, projDim/hDim>] of length 2
            .-> cat (Dim 1)      -- | <seqLen, 2*(oDim/hDim)>
          hLast = output                           -- | <seqLen, 2*(oDim/hDim)>
            .-> sliceDim 0 (seqLen-1) seqLen 1     -- | <1, 2*(oDim/hDim)>
            .-> (\o -> reshape (tail $ shape o) o) -- | <2*(oDim/hDim)>
      return (output, (hLast, cLast))
    else do -- the case of LSTM
      let h0c0f = (select 0 0 h0, select 0 0 c0) 
          (hsForward,csForward) = inputs
            .-> unstack          -- | [<iDim/oDim>] of length seqLen
            .-> scanl' (lstmCell singleLstmParams) h0c0f
            .-> tail             -- | [(<hDim>, <cDim>)] of length seqLen (by removing (h0,c0))
            .-> unzip            -- | ([<hDim>], [<cDim>])
          cLast = last csForward -- | <cDim>
            .-> singleton         -- | [<cDim>] of length 1
            .-> stack (Dim 0)    -- | <1, cDim>
            .-> (\o -> reshape (tail $ shape o) o) -- | <cDim>
          output = hsForward     -- | [<hDim>]
            .-> stack (Dim 0)    -- | <seqLen, hDim>
            .-> projLayer        -- | <seqLen, oDim/hDim>
          hLast = output                           -- | <seqLen, oDim/hDim>
            .-> sliceDim 0 (seqLen-1) seqLen 1     -- | <1, oDim/hDim>
            .-> (\o -> reshape (tail $ shape o) o) -- | <2*(oDim/hDim)>
      return (output, (hLast, cLast))

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
        d = if bidirectional then 2 else 1
        xh2Dim = (d * oDim) + hDim
    LstmParams
      <$> (SingleLstmParams
            <$> sample (LinearHypParams dev hasBias xh1Dim cDim) -- forgetGate
            <*> sample (LinearHypParams dev hasBias xh1Dim cDim) -- inputGate
            <*> sample (LinearHypParams dev hasBias xh1Dim cDim) -- candGate
            <*> sample (LinearHypParams dev hasBias xh1Dim hDim) -- outputGate
            <*> (sequence $ case projSize of 
                  Just projDim -> Just $ sample $ LinearHypParams dev hasBias hDim projDim
                  Nothing -> Nothing)
            )
      <*> forM [2..numLayers] (\_ ->
        SingleLstmParams 
          <$> sample (LinearHypParams dev hasBias xh2Dim cDim)
          <*> sample (LinearHypParams dev hasBias xh2Dim cDim)
          <*> sample (LinearHypParams dev hasBias xh2Dim cDim)
          <*> sample (LinearHypParams dev hasBias xh2Dim hDim)
          <*> (sequence $ case projSize of 
                Just projDim -> Just $ sample $ LinearHypParams dev hasBias hDim projDim
                Nothing -> Nothing)
          )

-- | The main function for LSTM layers
lstmLayers :: LstmParams -- ^ parameters (=model)
  -> Maybe Double    -- ^ introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.
  -> (Tensor,Tensor) -- ^ a pair of initial tensors: <D*numLayers,hDim>
  -> Tensor          -- ^ an input tensor <seqLen,iDim>
  -> (Tensor,(Tensor,Tensor)) -- ^ an output of (<seqLen,D*oDim>,(<D*numLayers,oDim>,<D*numLayers,cDim>))
lstmLayers LstmParams{..} dropoutProb (h0,c0) inputs = unsafePerformIO $ do
  let numLayers = length restLstmParams + 1
      (dnumLayers:(hiddenSize:_)) = shape h0
  unless (dnumLayers == numLayers * 2 || dnumLayers == numLayers) $ 
    ioError $ userError $ "illegal shape of h0: dnumLayers = " ++ (show dnumLayers) ++ "\nnumLayers = " ++ (show numLayers) 
  let bidirectional | dnumLayers == numLayers * 2 = True
                    | dnumLayers == numLayers = False
                    | otherwise = False -- Unexpected
      d = if bidirectional then 2 else 1
      (h0h:h0t) = [sliceDim 0 (d*i) (d*(i+1)) 1 h0 | i <- [0..numLayers]]
      (c0h:c0t) = [sliceDim 0 (d*i) (d*(i+1)) 1 c0 | i <- [0..numLayers]]
      firstLayer = singleLstmLayer bidirectional hiddenSize firstLstmParams (h0h,c0h) 
      restOfLayers = map (uncurry $ singleLstmLayer bidirectional hiddenSize) $ zip restLstmParams $ zip h0t c0t
      dropoutLayer = case dropoutProb of
                       Just prob -> unsafePerformIO . (dropout prob True)
                       Nothing -> id
      stackedLayers = \inputTensor -> 
                        scanr
                          (\nextLayer ohc -> nextLayer $ dropoutLayer $ fst ohc)
                          (firstLayer inputTensor) -- (<seqLen,D*oDim>,(<D*oDim>,<D*cDim>))
                          restOfLayers -- 
      -- | hncn:          [(<D*oDim>,<D*cDim>)] of length numLayers
      (outputList, hncn) = inputs -- | <seqLen,iDim> 
        .-> stackedLayers -- | [(<seqLen, D*oDim>,(<D*oDim>,<D*cDim>))] of length numLayers
        .-> unzip         -- | ([<seqLen, D*oDim>] of length numLayers, [(<D*oDim>,<D*cDim>)] of length numLayers)
      -- | unzip:         -> ([<D*oDim>] of length numLayers, [<D*cDim>] of length numLayers)
      -- | hn, cn:        -> [<D*oDim/cDim>] of length numLayers
      output = head outputList -- | [<seqLen, D*oDim>] of length numLayers
      -- | stack (Dim 0):  -> <numLayers, D*oDim/cDim>
      -- | reshape:        -> <D*numLayers, oDim/cDim>
      oDim = case projGate firstLstmParams of 
               Just linearParam -> 
                 let (o:_) = shape $ toDependent $ weight linearParam in o
               Nothing -> hiddenSize
      (hn, cn) = unzip hncn
      rh = \x -> reshape [d * numLayers, oDim] $ stack (Dim 0) x
      rc = \x -> reshape [d * numLayers, hiddenSize] $ stack (Dim 0) x
  return (output, (rh hn, rc cn))

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
