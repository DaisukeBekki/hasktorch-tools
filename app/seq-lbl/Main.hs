{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import GHC.Generics                   --base
import qualified Data.Text as T       --text
import qualified Data.List as L       --base
import qualified Data.Map.Strict as M --
--hasktorch
import Torch.Tensor (TensorLike(..),toCPU)
import Torch.TensorFactories (randnIO')
import Torch.Functional (mseLoss,add,matmul)
import Torch.NN         (Parameter,Parameterized,Randomizable,sample)
import Torch.Autograd   (IndependentTensor(..),makeIndependent)
import Torch.Optim      (GD(..))
import Torch.Train      (update,showLoss,zeroTensor,saveParams) --, loadParams)
import Torch.Control    (mapAccumM,foldLoop)
import Torch.Layer.Linear (LinearHypParams(..),linearLayer)
import Torch.Layer.LSTM (LSTMHypParams(..),LSTMParams,lstm)
import Torch.Util.Chart (drawLearningCurve)
import Torch.Util.Dict (oneHotFactory)

type Dat = T.Text
-- Imp | SoSo | NoImp deriving (Eq,Show,Enum,Bounded)

wrds :: [T.Text]
wrds = ["長野","は","日照時間","の","長さ","に","昼夜","の","温度差","と","果物作り","に","適した","土地","です","。"]

trainData :: [(Dat,Float)]
trainData = [
  ("長野",1),
  ("は",0),
  ("日照時間",1),
  ("の",0),
  ("長さ",0.5),
  ("に",0),
  ("昼夜",0),
  ("の",0),
  ("温度差",0.5),
  ("と",0),
  ("果物作り",1),
  ("に",0),
  ("適した",0.5),
  ("土地",0),
  ("です",0),
  ("。",0)
  ]

testData :: [(Dat,Float)]
testData = [
  ("長野",1),
  ("は",0),
  ("土地",0),
  ("です",0),
  ("。",0)
  ]

data HypParams = HypParams {
  lstmHypParams :: LSTMHypParams,
  wemb_input_dim :: Int,
  wemb_output_dim :: Int
  } deriving (Eq, Show)

data Params = Params {
  lstmParams :: LSTMParams,
  c0 :: Parameter,
  h0 :: Parameter,
  w_emb :: Parameter
  } deriving (Show, Generic)

instance Parameterized Params

instance Randomizable HypParams Params where
  sample HypParams{..} = do
    Params
      <$> sample lstmHypParams
      <*> (makeIndependent =<< randnIO' [wemb_output_dim])
      <*> (makeIndependent =<< randnIO' [wemb_output_dim])
      <*> (makeIndependent =<< randnIO' [wemb_input_dim, wemb_output_dim])

main :: IO()
main = do
  let iter = 50::Int
      (oneHotFor,wemb_dim) = oneHotFactory 0 wrds
  -- putStrLn $ show $ asTensor $ oneHotFor "土地"
  -- putStrLn $ show $ asTensor $ oneHotFor "の"
  initModel <- sample $ HypParams (LSTMHypParams 5 5) 5 wemb_dim
  let l = lstm (lstmParams initModel) (toDependent $ c0 initModel) (toDependent $ h0 initModel)
      i = map (asTensor . oneHotFor) ["土地","の","もの","が"]
  putStrLn $ show $ (toDependent $ w_emb initModel) `matmul` (asTensor $ oneHotFor "土地")
  putStrLn $ show $ l i
  {-
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let batchLoss = foldLoop trainData zeroTensor $ \(input,output) loss ->
                      let y' = linearLayer model $ toCPU $ asTensor input
                          y = toCPU $ asTensor output
                      in add loss $ mseLoss y y'
        lossValue = (asValue batchLoss)::Float
    showLoss 5 epoc lossValue
    u <- update model opt batchLoss 5e-4
    return (u, lossValue)
  --saveParams trainedModel "regression.model"
  --mapM_ (putStr . printf "%2.3f ") $ reverse allLosses
  drawLearningCurve "graph-seq.png" "Learning Curve" [("", reverse losses)]
  --loadedModel <- loadParams (Lin:@earHypParams 1 1) "regression.moxdel"
  --print loadedModel
  -}

