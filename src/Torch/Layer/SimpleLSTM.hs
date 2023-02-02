{-# LANGUAGE DeriveGeneric #-}

module Torch.Layer.SimpleLSTM (
  LstmHypParams(..)
  , LstmParams(..)
  , lstmLayers
  ) where 

import GHC.Generics              --base
import Torch.Tensor  (Tensor(..))
import Torch.Device  (Device(..))
import Torch.NN      (Parameterized(..),Randomizable(..),sample)

-- hasktorch-tools
import qualified Torch.Layer.LSTM as LSTM

data LstmHypParams = LstmHypParams {
  dev :: Device
  , bidirectional :: Bool 
  , input_size :: Int  -- ^ The number of expected features in the input x
  , hidden_size :: Int -- ^ The number of features in the hidden state h
  , num_layers :: Int     -- ^ Number of recurrent layers
  , dropoutProb :: Maybe Double  -- ^ If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.
  -- , bias :: Bool  -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  -- , batch_first :: Bool -- ^ If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
  , proj_size :: Maybe Int -- ^ If > 0, will use LSTM with projections of corresponding size.
  } deriving (Eq, Show)

data LstmParams = LstmParams {
  lstmParams :: LSTM.LstmParams
  , initialStatesParams :: LSTM.InitialStatesParams
  } deriving (Show, Generic)
instance Parameterized LstmParams

instance Randomizable LstmHypParams LstmParams where
  sample LstmHypParams{..} = 
    LstmParams
      <$> (sample $ LSTM.LstmHypParams dev bidirectional input_size hidden_size num_layers dropoutProb proj_size)
      <*> (sample $ LSTM.InitialStatesHypParams dev bidirectional hidden_size num_layers)

lstmLayers :: LstmHypParams -- ^ hyper params
  -> LstmParams      -- ^ params
  -> Tensor          -- ^ an input tensor of shape (L,H_in)
  -> IO Tensor       -- ^ [his] of shape (L,D*H_out)
lstmLayers LstmHypParams{..} LstmParams{..} = 
  LSTM.lstmLayers 
    (LSTM.LstmHypParams dev bidirectional input_size hidden_size num_layers dropoutProb proj_size)
    lstmParams
    (LSTM.c0h0s initialStatesParams)
