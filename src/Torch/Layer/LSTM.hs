{-# LANGUAGE DeriveGeneric #-}

module Torch.Layer.LSTM (
  LstmHypParams(..)
  , LstmParams(..)
  , lstmLayers
  ) where 

import Prelude hiding (tanh) 
import GHC.Generics              --base
import Data.Maybe (isJust)    --base
import Data.List (scanl',foldl',scanr) --base
import Control.Monad (forM,unless)      --base
--hasktorch
import Torch.Tensor       (Tensor(..),shape,select,sliceDim)
import Torch.Functional   (Dim(..),sigmoid,cat,stack,dropout)
import Torch.Device       (Device(..))
import Torch.NN           (Parameterized(..),Randomizable(..),Parameter,sample)
import Torch.Autograd (IndependentTensor(..),makeIndependent)
import Torch.Tensor.Util (unstack)
import Torch.Tensor.TensorFactories (randnIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)

data LstmHypParams = LstmHypParams {
  dev :: Device
  , bidirectional :: Bool 
  , input_size :: Int  -- ^ The number of expected features in the input x
  , hidden_size :: Int -- ^ The number of features in the hidden state h
  , num_layers :: Int     -- ^ Number of recurrent layers
  , dropoutProb :: Maybe Double  -- ^ If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.
  -- , bias :: Bool  -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  -- , batch_first :: Bool -- ^ If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
  -- , proj_size :: Float -- ^ If > 0, will use LSTM with projections of corresponding size.
  } deriving (Eq, Show)

data SingleLstmParams = SingleLstmParams {
    forgetGate :: LinearParams,
    inputGate :: LinearParams,
    candidateGate :: LinearParams,
    outputGate :: LinearParams
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

data LstmParams = LstmParams {
  firstLstmParams :: SingleLstmParams -- ^ a model for the first LSTM layer
  , restLstmParams :: [SingleLstmParams] -- ^ models for the rest of LSTM layers
  } deriving (Show, Generic)
instance Parameterized LstmParams

-- (makeIndependent =<< randnIO' dev [c_Dim]) 
instance Randomizable LstmHypParams LstmParams where
  sample LstmHypParams{..} = do
    let x_Dim = input_size
        h_Dim = hidden_size
        c_Dim = hidden_size
        xh1_Dim = x_Dim + h_Dim
        xh2_Dim = if bidirectional
                    then (2 * x_Dim) + h_Dim
                    else xh1_Dim
    LstmParams
      <$> (SingleLstmParams
            <$> sample (LinearHypParams dev xh1_Dim c_Dim)
            <*> sample (LinearHypParams dev xh1_Dim c_Dim)
            <*> sample (LinearHypParams dev xh1_Dim c_Dim)
            <*> sample (LinearHypParams dev xh1_Dim h_Dim)
            )
      <*> forM [2..num_layers] (\_ ->
        SingleLstmParams 
          <$> sample (LinearHypParams dev xh2_Dim c_Dim)
          <*> sample (LinearHypParams dev xh2_Dim c_Dim)
          <*> sample (LinearHypParams dev xh2_Dim c_Dim)
          <*> sample (LinearHypParams dev xh2_Dim h_Dim)
          )

-- | inputのlistから、(cellState,hiddenState=output)のリストを返す
-- | scanl' :: ((h,c) -> input -> (h',c')) -> (h0,c0) -> [input] -> [(hi,ci)]
singleLstmLayer :: Bool -- ^ True if BiLSTM, False otherwise
  -> Int                -- ^ hidden_size 
  -> SingleLstmParams   -- ^ params
  -> (Tensor,Tensor)    -- ^ A pair (h0,c0) of shape (hidden_size) or (2,hidden_size)
  -> [Tensor]           -- ^ an input layer
  -> IO [Tensor]        -- ^ [forward_hi] or [forward_hi+backward_hi]
singleLstmLayer isBiLSTM stateDim params (h0,c0) inputs = do
  let h0shape = shape h0
      c0shape = shape c0
  if isBiLSTM -- check the well-formedness of the shapes of h0 and c0
    then do -- BiLSTM
      unless ((h0shape == [2,stateDim]) && (c0shape == [2,stateDim])) $
        ioError $ userError $ "illegal h0 or c0 shape: " ++ (show h0shape) ++ " or " ++ (show c0shape)
      let h0f = select 0 0 h0
          h0b = select 0 1 h0
          c0f = select 0 0 c0
          c0b = select 0 1 c0
          -- | 以下、(c0,h0)は除くためtailを取る
          forwardLayer = fst $ unzip $ tail $ scanl' (lstmCell params) (h0f,c0f) inputs 
          backwardLayer = fst $ unzip $ init $ scanr (flip $ lstmCell params) (h0b,c0b) inputs 
      return $ map (\(f,b)-> cat (Dim 0) [f,b]) $ zip forwardLayer backwardLayer
    else do -- LSTM
      unless ((h0shape == [stateDim]) && (c0shape == [stateDim])) $
        ioError $ userError $ "illegal h0 or c0 shape: " ++ (show h0shape) ++ " or " ++ (show c0shape)
      return $ fst $ unzip $ tail $ scanl' (lstmCell params) (h0,c0) inputs -- | (c0,h0)は除くためtailを取る

lstmLayers :: LstmHypParams -- ^ hyper params
  -> LstmParams      -- ^ params
  -> Tensor          -- ^ an input tensor of shape (L,H_in)
  -> (Tensor,Tensor) -- ^ a pair of initial tensors: (D*num_layers,H_out)
  -> IO Tensor       -- ^ [his] of shape (L,D*H_out)
lstmLayers LstmHypParams{..} LstmParams{..} inputs (h0,c0) = do
  let (h0h:h0t) = if bidirectional
                    then [sliceDim 0 (2*i) (2*i+2) 1 h0 | i <- [0..num_layers]]
                    else [sliceDim 0 i (i+1) 1 h0 | i <- [0..num_layers]]
      (c0h:c0t) = if bidirectional
                    then [sliceDim 0 (2*i) (2*i+2) 1 c0 | i <- [0..num_layers]]
                    else [sliceDim 0 i (i+1) 1 c0 | i <- [0..num_layers]]
  firstLayer <- singleLstmLayer bidirectional hidden_size firstLstmParams (h0h,c0h) $ unstack inputs
  let restOfLayers = map (uncurry $ singleLstmLayer bidirectional hidden_size) $ zip restLstmParams $ zip h0t c0t
      dropoutLayer = \hs -> case dropoutProb of
                              Just prob -> dropout prob True hs
                              Nothing -> return hs
  foldl' (\hs nextLayer -> 
           stack (Dim 0) <$> (nextLayer =<< (unstack <$> (dropoutLayer =<< hs))))
         (return $ stack (Dim 0) firstLayer)
         restOfLayers

