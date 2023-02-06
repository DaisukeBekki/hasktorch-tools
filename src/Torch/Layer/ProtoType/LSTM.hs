{-# LANGUAGE DeriveGeneric #-}

module Torch.Layer.ProtoType.LSTM (
  LstmHypParams(..),
  LstmParams(..),
  lstmCell,
  lstmLayers
  ) where 

import Prelude hiding (tanh) 
import GHC.Generics              --base
import Data.List (scanl',foldl') --base
import Control.Monad (forM)      --base
--hasktorch
import Torch.Tensor       (Tensor(..))
import Torch.Functional   (Dim(..),sigmoid,cat,stack)
import Torch.Device       (Device(..))
import Torch.NN           (Parameterized(..),Randomizable(..),Parameter,sample)
import Torch.Autograd (IndependentTensor(..),makeIndependent)
import Torch.Tensor.Util (unstack)
import Torch.Tensor.TensorFactories (randnIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)

data LstmHypParams = LstmHypParams {
  dev :: Device,
  stateDim :: Int,
  numOfLayers :: Int
  } deriving (Eq, Show)

data SingleLstmParams = SingleLstmParams {
    forgetGate :: LinearParams,
    inputGate :: LinearParams,
    candidateGate :: LinearParams,
    outputGate :: LinearParams
    } deriving (Show, Generic)
instance Parameterized SingleLstmParams

data LstmParams = LstmParams {
  lstmParams :: [SingleLstmParams],
  initialParams :: [(Parameter,Parameter)]
  } deriving (Show, Generic)

instance Parameterized LstmParams
instance Randomizable LstmHypParams LstmParams where
  sample LstmHypParams{..} = do
    let x_Dim = stateDim
        h_Dim = stateDim
        c_Dim = stateDim
        xh_Dim = x_Dim + h_Dim
    LstmParams
      <$> forM [1..numOfLayers] (\_ ->
        SingleLstmParams 
          <$> sample (LinearHypParams dev True xh_Dim c_Dim)
          <*> sample (LinearHypParams dev True xh_Dim c_Dim)
          <*> sample (LinearHypParams dev True xh_Dim c_Dim)
          <*> sample (LinearHypParams dev True xh_Dim h_Dim)
          )
      <*> forM [1..(numOfLayers-1)] (\_ ->
        (\x y -> (x,y)) -- IO(a)->IO(b->(a,b))
          <$> (makeIndependent =<< randnIO' dev [c_Dim]) 
          <*> (makeIndependent =<< randnIO' dev [x_Dim])
          )

lstmCell :: SingleLstmParams 
  -> (Tensor,Tensor) -- ^ (ct,ht) 
  -> Tensor          -- ^ xt
  -> (Tensor,Tensor) -- ^ (ct',ht')
lstmCell SingleLstmParams{..} (ht,ct) xt =
  let xt_ht = cat (Dim 0) [xt,ht]
      ft = sigmoid $ linearLayer forgetGate $ xt_ht
      it = sigmoid $ linearLayer inputGate $ xt_ht
      cant = tanh' $ linearLayer candidateGate $ xt_ht
      ct' = (ft * ct) + (it * cant)
      ot = sigmoid $ linearLayer outputGate $ xt_ht
      ht' = ot * (tanh' ct')
  in (ht', ct')
  where -- | HACK : Torch.Functional.tanhとexpが `Segmentation fault`になるため
    tanh' x = 2 * (sigmoid $ 2 * x) - 1

-- | inputのlistから、(cellState,hiddenState=output)のリストを返す
-- | scanl' :: ((h,c) -> input -> (h',c')) -> (h0,c0) -> [input] -> [(hi,ci)]
lstmLayer :: Bool       -- ^ if BiLSTM then True else False
  -> SingleLstmParams   -- ^ hyper params
  -> (Tensor,Tensor)    -- ^ A pair of vectors (h0,c0)
  -> [Tensor]           -- ^ an input layer
  -> [(Tensor,Tensor)]  -- ^ the list of (hi,ci) pairs
lstmLayer ifBiLstm params (h0,c0) inputs = 
  let firstLayer = tail $ scanl' (lstmCell params) (h0,c0) inputs in
    if ifBiLstm -- | (c0,h0)は除くためtailを取る
      then reverse $ tail $ scanl' (lstmCell params) (last firstLayer) $ reverse $ snd $ unzip firstLayer
      else firstLayer

lstmLayers :: Bool     -- ^ True if BiLSTM, False otherwise
  -> LstmParams        -- ^ a model
  -> (Tensor,Tensor)   -- ^ a pair of initial tensors (h0,c0)
  -> [Tensor]          -- ^ an input layer
  -> [(Tensor,Tensor)] -- ^ the list of (hi,ci)
lstmLayers ifBiLstm LstmParams{..} h0c0 inputs = 
  let initialTensors = map (\(h,c) -> (toDependent h,toDependent c)) initialParams 
      (firstParams:restParams) = lstmParams
      firstLayer = lstmLayer ifBiLstm firstParams h0c0 inputs in
  foldl' (\hicis nextLayer -> nextLayer $ snd $ unzip hicis)
         firstLayer
         (map (uncurry $ lstmLayer ifBiLstm) (zip restParams initialTensors))
