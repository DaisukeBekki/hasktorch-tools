{-# LANGUAGE ExtendedDefaultRules, DeriveGeneric #-}

module Torch.LangModel.Juman.Model (
  JLstmHypParams(..),
  JLstmParams(..),
  jLstmLayer
  ) where

--import Control.Monad (when)        --base
import qualified GHC.Generics as G --base
--import qualified Data.Aeson as A   --aeson
import qualified Data.Text as T    --text
import qualified Text.Juman as J   --juman-tools
-- | hasktorch
import Torch.Tensor (Tensor(..))
import Torch.Functional (Dim(..),stack)
import Torch.Device (Device(..),DeviceType(..))
import Torch.NN (Parameterized(..),Randomizable(..),sample)
-- | hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer) 
import Torch.Layer.MLP (MLPHypParams(..),MLPParams(..),ActName(..),mlpLayer)
import Torch.Layer.SimpleLSTM as L (LstmHypParams(..),LstmParams(..),lstmLayers)
import Torch.LangModel.Juman.Dict (WordInfo,jumanData2Tuples) 
import Torch.Tensor.Util (unstack)

-- | hyper parameters：ここで必要な
data JLstmHypParams = JLstmHypParams {
  lstmHypParams :: LstmHypParams
  , dictDim :: Int
  , mlpLayers :: [(Int,ActName)]
  } deriving (Eq, Show)

data JLstmParams = JLstmParams {
  wEmbParams :: LinearParams
  , lstmParams :: LstmParams
  , mlpParams :: MLPParams
  } deriving (G.Generic)
instance Parameterized JLstmParams

instance Randomizable JLstmHypParams JLstmParams where
  sample JLstmHypParams{..} = 
    case proj_size lstmHypParams of
      Just proj -> JLstmParams
        <$> (sample $ LinearHypParams (L.dev lstmHypParams) True dictDim (input_size lstmHypParams))
        <*> (sample lstmHypParams)
        <*> (sample $ MLPHypParams (L.dev lstmHypParams) proj mlpLayers)

jLstmLayer :: JLstmHypParams
  -> JLstmParams
  -> (WordInfo -> [Float]) -- ^ one-hot function
  -> T.Text                -- ^ input text
  -> IO ([Tensor],[T.Text])  -- ^ (output layer, surface forms)
jLstmLayer JLstmHypParams{..} JLstmParams{..} oneHot text = do
  jumanOutput <- J.text2jumanData' text -- [JumanData]
  let dev = L.dev lstmHypParams
      word_infos = jumanData2Tuples jumanOutput
      w_emb_layer = map (\w -> linearLayer wEmbParams $ asTensor'' dev $ oneHot w)
      lstm_layers = lstmLayers lstmHypParams lstmParams 
      mlp_layer = mlpLayer mlpParams 
  return $ ((unstack . mlp_layer . lstm_layers . (stack (Dim 0)) . w_emb_layer) word_infos,
           map (\(surfaceForm,_,_) -> surfaceForm) word_infos)

main :: IO ()
main = do
  let jLstmHypParams = JLstmHypParams {
        lstmHypParams = LstmHypParams {
          dev = Device CUDA 0
          , bidirectional = True
          , input_size = 10
          , hidden_size = 32
          , num_layers = 3
          , proj_size = Just 2
          },
        dictDim = 0,
        mlpLayers = [(1,Relu),(10,Relu)]
        }
  print "test done."

