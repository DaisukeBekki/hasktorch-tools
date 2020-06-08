{-# LANGUAGE DeriveGeneric #-}

module Torch.Layer.LSTM where 

import Prelude hiding (tanh) 
import GHC.Generics       --base
import Data.List (scanl') --base
                          --hasktorch
import Torch.Tensor       (Tensor(..))
import Torch.Functional   (sigmoid,tanh,cat)
import Torch.NN           (Parameterized,Randomizable,sample)
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)

data LSTMHypParams = LSTMHypParams {
  inputDim :: Int,
  outputDim :: Int
  } deriving (Eq, Show)

data LSTMParams = LSTMParams {
  forgetGate :: LinearParams,
  inputGate :: LinearParams,
  outputGate :: LinearParams
  } deriving (Show, Generic)

instance Parameterized LSTMParams

instance Randomizable LSTMHypParams LSTMParams where
  sample LSTMHypParams{..} = do
    let hiddenStateDim = outputDim
        cellStateDim = hiddenStateDim + inputDim 
    LSTMParams
      <$> sample (LinearHypParams cellStateDim cellStateDim)
      <*> sample (LinearHypParams cellStateDim cellStateDim)
      <*> sample (LinearHypParams cellStateDim cellStateDim)

lstmCell :: LSTMParams -> (Tensor,Tensor) -> Tensor -> (Tensor,Tensor)
lstmCell LSTMParams{..} (ct,ht) xt =
  let xt_ht = cat 0 [xt,ht]
      ct' = ct * (sigmoid $ linearLayer forgetGate $ xt_ht)
            + (tanh xt_ht) * (sigmoid $ linearLayer inputGate $ xt_ht)
      ht' = (tanh ct') * (sigmoid $ linearLayer outputGate $ xt_ht)
  in (ct', ht')

-- | inputのlistから、(cellState,hiddenState=output)のリストを返す
-- | scanl' :: ((c,h) -> input -> (c',h')) -> (c0,h0) -> [input] -> [(ci,hi)]
lstm :: LSTMParams -> (Tensor,Tensor) -> [Tensor] -> [(Tensor,Tensor)]
lstm params (cellState,hiddenState) inputs = scanl' (lstmCell params) (cellState,hiddenState) inputs

-- squeezeAll??
