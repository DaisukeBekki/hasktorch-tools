{-# LANGUAGE DeriveGeneric #-}

{-
h0,c0をLSTMのモデルに含んでしまうバージョン
h0,c0を別途初期化するのが面倒な場合はこちらを使用する。
Torch.Layer.LSTMの関数を呼び出して実装しているので内部の設計は同じ。
-}

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
  , inputSize :: Int  -- ^ The number of expected features in the input x
  , hiddenSize :: Int -- ^ The number of features in the hidden state h
  , numLayers :: Int     -- ^ Number of recurrent layers
  , hasBias :: Bool  -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  -- , batch_first :: Bool -- ^ If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
  , projSize :: Maybe Int -- ^ If > 0, will use LSTM with projections of corresponding size.
  } deriving (Eq, Show)

data LstmParams = LstmParams {
  lstmParams :: LSTM.LstmParams
  , initialStatesParams :: LSTM.InitialStatesParams
  } deriving (Show, Generic)
instance Parameterized LstmParams

instance Randomizable LstmHypParams LstmParams where
  sample LstmHypParams{..} = 
    LstmParams
      <$> (sample $ LSTM.LstmHypParams dev bidirectional inputSize hiddenSize numLayers hasBias projSize)
      <*> (sample $ LSTM.InitialStatesHypParams dev bidirectional hiddenSize numLayers)

lstmLayers :: LstmParams -- ^ params
  -> Maybe Double -- ^ dropout
  -> Tensor       -- ^ an input tensor of shape (L,H_in)
  -> Tensor       -- ^ [his] of shape (L,D*H_out)
lstmLayers LstmParams{..} = 
  LSTM.lstmLayers lstmParams (LSTM.toDependentTensors initialStatesParams) 
