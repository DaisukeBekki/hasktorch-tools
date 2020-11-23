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
import Torch.Tensor (Tensor(..),TensorLike(..),toCPU,reshape)
import Torch.TensorFactories (randnIO')
import Torch.Functional (Dim(..),cat,logSoftmax,mseLoss,add,matmul,nllLoss')
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

data Label = Important | SoSo | NotImportant deriving (Eq,Show,Enum,Bounded)

wrds :: [T.Text]
wrds = ["長野","は","日照時間","の","長さ","に","昼夜","の","温度差","と","果物作り","に","適した","土地","です","。"]

trainData :: [(Dat,Label)]
trainData = [
  ("長野",Important),
  ("は",NotImportant),
  ("日照時間",Important),
  ("の",NotImportant),
  ("長さ",SoSo),
  ("に",NotImportant),
  ("昼夜",NotImportant),
  ("の",NotImportant),
  ("温度差",SoSo),
  ("と",NotImportant),
  ("果物作り",Important),
  ("に",NotImportant),
  ("適した",SoSo),
  ("土地",NotImportant),
  ("です",NotImportant),
  ("。",NotImportant)
  ]

testData :: [(Dat,Label)]
testData = [
  ("長野",Important),
  ("は",NotImportant),
  ("果物作り",Important),
  ("に",NotImportant),
  ("適した",SoSo),
  ("場所",SoSo),
  ("です",NotImportant),
  ("。",NotImportant)
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
      <*> sample (LinearHypParams (stateDim lstmHypParams) 3)

main :: IO()
main = do
  let iter = 200::Int
      lstm_dim = 128
      learningRate = 5e-4
      graphFileName = "graph-seq-class.png"
      modelFileName = "seq-class.model"
      (oneHotFor,wemb_dim) = oneHotFactory 0 wrds
      hyperParams = HypParams (LSTMHypParams lstm_dim) wemb_dim
  initModel <- sample hyperParams
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let (_,_,batchLoss) = feedForward model oneHotFor trainData 
        lossValue = (asValue batchLoss)::Float
    showLoss 5 epoc lossValue
    u <- update model opt batchLoss learningRate
    return (u, lossValue)
  saveParams trainedModel modelFileName
  drawLearningCurve graphFileName "Learning Curve" [("", reverse losses)]
  --loadedModel <- loadParams hyperParams modelFileName
  let (y',y,loss) = feedForward trainedModel oneHotFor testData
  putStrLn $ show y'
  putStrLn $ show $ map (\v -> (toEnum v)::Label) $ asValue y
  putStrLn $ show $ ((asValue loss)::Float)

-- | returns (ys', ys, batchLoss) i.e. (predictions, groundtruths, batchloss)
feedForward :: Params -> (T.Text -> [Float]) -> [(Dat,Label)] -> (Tensor,Tensor,Tensor)
feedForward model oneHotFor dataSet = 
  let lstm = bilstmLayer (lstmParams model) (toDependent $ c0 model) (toDependent $ h0 model)
      embLayer = map (\w -> (toDependent $ w_emb model) `matmul` (asTensor $ oneHotFor w)) $ fst $ unzip $ dataSet
      y' = cat (Dim 0) $ map (reshape [1,3] . logSoftmax (Dim 0) . linearLayer (mlpParams model)) $ fst $ unzip $ lstm embLayer
      y  = cat (Dim 0) $ map (reshape [1] . asTensor . fromEnum) $ snd $ unzip $ dataSet
  in (y', y, nllLoss' y y')

