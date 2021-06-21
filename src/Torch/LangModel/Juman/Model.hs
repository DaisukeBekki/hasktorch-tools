{-# LANGUAGE ExtendedDefaultRules, DeriveGeneric #-}

module Torch.LangModel.Juman.Model (
  JLSTMHypParams(..),
  JLSTMParams(..),
  jLSTMlayer
  ) where

--import Control.Monad (when)        --base
import qualified GHC.Generics as G --base
--import qualified Data.Aeson as A   --aeson
import qualified Data.Text as T    --text
import qualified Text.Juman as J   --juman-tools
-- | hasktorch
import Torch.Tensor (Tensor(..))
import Torch.Device (Device(..))
import Torch.NN (Parameterized(..),Randomizable(..),Parameter,sample)
import Torch.Autograd (IndependentTensor(..),makeIndependent)
-- | hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'',randnIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.MLP (MLPHypParams(..),MLPParams(..),ActName(..))
import Torch.Layer.BiLSTM (BiLstmHypParams(..),BiLstmParams(..),biLstmLayers)
import Torch.LangModel.Juman.Dict (WordInfo,jumanData2Tuples) 

data JLSTMHypParams = JLSTMHypParams {
  dev :: Device,
  dictDim :: Int,
  embDim :: Int,
  mlp_layers :: [(Int,ActName)]
  } deriving (Eq, Show)

data JLSTMParams = JLSTMParams {
  w_emb :: LinearParams, 
  c_0 :: Parameter,
  h_0 :: Parameter,
  bilstm_params :: BiLstmParams,
  mlp_params :: MLPParams
  } deriving (G.Generic)

instance Parameterized JLSTMParams

instance Randomizable JLSTMHypParams JLSTMParams where
  sample JLSTMHypParams{..} = 
    JLSTMParams
    <$> sample (LinearHypParams dev dictDim embDim)
    <*> (makeIndependent =<< randnIO' dev [embDim])
    <*> (makeIndependent =<< randnIO' dev [embDim])
    <*> sample (BiLstmHypParams dev 1 embDim)
    <*> sample (MLPHypParams dev (embDim * 2) mlp_layers)

jLSTMlayer :: JLSTMParams
           -> Device                -- ^ device for tensors
           -> (WordInfo -> [Float]) -- ^ one-hot function
           -> T.Text                -- ^ input text
           -> IO([Tensor], [T.Text])  -- ^ (output layer, surface forms)
jLSTMlayer JLSTMParams{..} dev oneHot text = do
  jumanOutput <- J.text2jumanData' text -- [JumanData]
  let wordInfos = jumanData2Tuples jumanOutput
      w_emb_layer = map (\w -> linearLayer w_emb $ asTensor'' dev $ oneHot w) wordInfos
      output_layer = snd $ unzip $ biLstmLayers bilstm_params (toDependent c_0,toDependent h_0) w_emb_layer
  return $ (output_layer,
            map (\(surfaceForm,_,_) -> surfaceForm) wordInfos)

