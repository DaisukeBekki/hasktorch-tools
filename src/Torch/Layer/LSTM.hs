{-# LANGUAGE DeriveGeneric #-}

module Torch.Layer.LSTM (
  LSTMHypParams(..),
  LSTMParams(..),
  lstmLayer,
  bilstmLayer
  ) where 

import Prelude hiding (tanh) 
import GHC.Generics       --base
import Data.List (scanl') --base
--hasktorch
import Torch.Tensor       (Tensor(..))
import Torch.Functional   (Dim(..),sigmoid,tanh,cat)
import Torch.NN           (Parameterized,Randomizable,sample)
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)

data LSTMHypParams = LSTMHypParams {
  stateDim :: Int
  } deriving (Eq, Show)

data LSTMParams = LSTMParams {
  forgetGate :: LinearParams,
  inputGate :: LinearParams,
  candidateGate :: LinearParams,
  outputGate :: LinearParams
  } deriving (Show, Generic)

instance Parameterized LSTMParams

instance Randomizable LSTMHypParams LSTMParams where
  sample LSTMHypParams{..} = do
    let x_Dim = stateDim
        h_Dim = stateDim
        c_Dim = stateDim
        xh_Dim = x_Dim + h_Dim
    LSTMParams
      <$> sample (LinearHypParams xh_Dim c_Dim)
      <*> sample (LinearHypParams xh_Dim c_Dim)
      <*> sample (LinearHypParams xh_Dim c_Dim)
      <*> sample (LinearHypParams xh_Dim h_Dim)

lstmCell :: LSTMParams -> (Tensor,Tensor) -> Tensor -> (Tensor,Tensor)
lstmCell LSTMParams{..} (ct,ht) xt =
  let xt_ht = cat (Dim 0) [xt,ht]
      ft = sigmoid $ linearLayer forgetGate $ xt_ht
      it = sigmoid $ linearLayer inputGate $ xt_ht
      cant = tanh $ linearLayer candidateGate $ xt_ht
      ct' = (ft * ct) + (it * cant)
      ot = sigmoid $ linearLayer outputGate $ xt_ht
      ht' = ot * (tanh ct')
  in (ct', ht')

-- | inputのlistから、(cellState,hiddenState=output)のリストを返す
-- | scanl' :: ((c,h) -> input -> (c',h')) -> (c0,h0) -> [input] -> [(ci,hi)]
lstmLayer :: LSTMParams -> Tensor -> Tensor -> [Tensor] -> [(Tensor,Tensor)]
lstmLayer params c0 h0 inputs = scanl' (lstmCell params) (c0,h0) inputs

bilstmLayer :: LSTMParams -> Tensor -> Tensor -> [Tensor] -> [(Tensor,Tensor)]
bilstmLayer params c0 h0 inputs =
  let firstLayer = scanl' (lstmCell params) (c0,h0) inputs in
  reverse $ scanl' (lstmCell params) (last firstLayer) $ reverse $ snd $ unzip firstLayer

