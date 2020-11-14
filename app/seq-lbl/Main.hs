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
import Torch.Layer.Linear (LinearHypParams(..),LinearParams,linearLayer)
import Torch.Layer.LSTM (LSTMHypParams(..),LSTMParams,lstmLayer,bilstmLayer)
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
  wemb_dim :: Int
  } deriving (Eq, Show)

data Params = Params {
  lstmParams :: LSTMParams,
  c0 :: Parameter,
  h0 :: Parameter,
  w_emb :: Parameter,
  mlpParams :: LinearParams
  } deriving (Show, Generic)

instance Parameterized Params

instance Randomizable HypParams Params where
  sample HypParams{..} = do
    Params
      <$> sample lstmHypParams
      <*> (makeIndependent =<< randnIO' [stateDim lstmHypParams])
      <*> (makeIndependent =<< randnIO' [stateDim lstmHypParams])
      <*> (makeIndependent =<< randnIO' [stateDim lstmHypParams, wemb_dim])
      <*> sample (LinearHypParams (stateDim lstmHypParams) 1)

main :: IO()
main = do
  let iter = 3000::Int
      lstm_dim = 2
      learningRate = 5e-4
      graphFileName = "graph-seq.png"
      modelFileName = "seq.model"
      (oneHotFor,wemb_dim) = oneHotFactory 0 wrds
      hyperParams = HypParams (LSTMHypParams lstm_dim) wemb_dim
  initModel <- sample hyperParams
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let lstm = bilstmLayer (lstmParams model) (toDependent $ c0 model) (toDependent $ h0 model)
        embLayer = map (\w -> (toDependent $ w_emb model) `matmul` (asTensor $ oneHotFor w)) $ fst $ unzip $ trainData
        ys' = map (linearLayer (mlpParams model)) $ fst $ unzip $ lstm embLayer
        ys  = map asTensor $ snd $ unzip $ trainData
        batchLoss = foldLoop (zip ys' ys) zeroTensor $ \(y',y) loss ->
                      add loss $ mseLoss y y'
        lossValue = (asValue batchLoss)::Float
    showLoss 5 epoc lossValue
    u <- update model opt batchLoss learningRate
    return (u, lossValue)
  --saveParams trainedModel modelFileName
  --mapM_ (putStr . printf "%2.3f ") $ reverse allLosses
  drawLearningCurve graphFileName "Learning Curve" [("", reverse losses)]
  --loadedModel <- loadParams hyperParams modelFileName
  --print loadedModel

