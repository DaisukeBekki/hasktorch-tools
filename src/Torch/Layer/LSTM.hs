{-# LANGUAGE DeriveGeneric #-}

module Torch.Layer.LSTM (
  LstmHypParams(..),
  LstmParams(..),
  lstmCell,
  lstmLayer,
  lstmLayer'
  ) where 

import Prelude hiding (tanh) 
import GHC.Generics       --base
import Data.List (scanl') --base
--hasktorch
import Torch.Tensor       (Tensor(..))
import Torch.Functional   (Dim(..),sigmoid,tanh,cat)
import Torch.Device       (Device(..))
import Torch.NN           (Parameterized,Randomizable,sample)
import Torch.Tensor.Util (unstack)
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)

data LstmHypParams = LstmHypParams {
  dev :: Device,
  stateDim :: Int
  } deriving (Eq, Show)

data LstmParams = LstmParams {
  forgetGate :: LinearParams,
  inputGate :: LinearParams,
  candidateGate :: LinearParams,
  outputGate :: LinearParams
  } deriving (Show, Generic)

instance Parameterized LstmParams

instance Randomizable LstmHypParams LstmParams where
  sample LstmHypParams{..} = do
    let x_Dim = stateDim
        h_Dim = stateDim
        c_Dim = stateDim
        xh_Dim = x_Dim + h_Dim
    LstmParams
      <$> sample (LinearHypParams dev xh_Dim c_Dim)
      <*> sample (LinearHypParams dev xh_Dim c_Dim)
      <*> sample (LinearHypParams dev xh_Dim c_Dim)
      <*> sample (LinearHypParams dev xh_Dim h_Dim)

lstmCell :: LstmParams -> (Tensor,Tensor) -> Tensor -> (Tensor,Tensor)
lstmCell LstmParams{..} (ct,ht) xt =
  let xt_ht = cat (Dim 0) [xt,ht]
      ft = sigmoid $ linearLayer forgetGate $ xt_ht
      it = sigmoid $ linearLayer inputGate $ xt_ht
      cant = tanh' $ linearLayer candidateGate $ xt_ht
      ct' = (ft * ct) + (it * cant)
      ot = sigmoid $ linearLayer outputGate $ xt_ht
      ht' = ot * (tanh' ct')
  in (ct', ht')

-- | HACK : Torch.Functional.tanhとexpが `Segmentation fault`になるため
tanh' :: Tensor -> Tensor
tanh' x = 2 * (sigmoid $ 2 * x) - 1

-- | inputのlistから、(cellState,hiddenState=output)のリストを返す
-- | scanl' :: ((c,h) -> input -> (c',h')) -> (c0,h0) -> [input] -> [(ci,hi)]
lstmLayer :: LstmParams -- ^ hyper params
  -> (Tensor,Tensor) -- ^ A pair of vectors (c0,h0)
  -> [Tensor]        -- ^ input layer
  -> [(Tensor,Tensor)]
lstmLayer params (c0,h0) inputs = tail $ scanl' (lstmCell params) (c0,h0) inputs

lstmLayer' :: LstmParams -- ^ hyper params
  -> (Tensor,Tensor) -- ^ A pair of vectors (c0,h0)
  -> Tensor        -- ^ input layer of the shape (seq_length:embed_dim)
  -> [(Tensor,Tensor)]
lstmLayer' params (c0,h0) inputs = 
  tail $ scanl' (lstmCell params) (c0,h0) (unstack inputs)