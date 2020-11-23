{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import GHC.Generics                   --base
import qualified Data.Text as T       --text
import qualified Data.Text.IO as T       --text
--hasktorch
import Torch.Tensor       (Tensor(..),TensorLike(..),toCPU,reshape)
import Torch.TensorFactories (randnIO')
import Torch.Functional   (Dim(..),sigmoid,binaryCrossEntropyLoss',matmul,cat,squeezeAll)
import Torch.NN           (Parameter,Parameterized,Randomizable,sample)
import Torch.Autograd     (IndependentTensor(..),makeIndependent)
import Torch.Optim        (GD(..))
import Torch.Train        (update,showLoss,saveParams,loadParams) --, loadParams)
import Torch.Control      (mapAccumM)
import Torch.Layer.Linear (LinearHypParams(..),LinearParams,linearLayer)
import Torch.Layer.LSTM   (LSTMHypParams(..),LSTMParams,bilstmLayer)
import Torch.Util.Chart   (drawLearningCurve)
import Torch.Util.Dict    (oneHotFactory)

type Dat = T.Text

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
  ("果物作り",1),
  ("に",0),
  ("適した",0.5),
  ("地域",0),
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
  let iter = 2000::Int
      lstm_dim = 64
      (oneHotFor,wemb_dim) = oneHotFactory 0 wrds
      hyperParams = HypParams (LSTMHypParams lstm_dim) wemb_dim
      learningRate = 5e-2
      graphFileName = "graph-seq-reg.png"
      modelFileName = "seq-reg.model"
  -- | Training
  initModel <- sample hyperParams
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let (_,_,batchLoss) = feedForward model oneHotFor toCPU trainData
        lossValue = (asValue batchLoss)::Float
    showLoss 5 epoc lossValue
    u <- update model opt batchLoss learningRate
    return (u, lossValue)
  saveParams trainedModel modelFileName
  drawLearningCurve graphFileName "Learning Curve" [("", reverse losses)]
  -- | Test
  loadedModel <- loadParams hyperParams modelFileName
  let (y,_,_) = feedForward loadedModel oneHotFor toCPU testData
  putStrLn "\nPredictions:"
  --T.putStrLn $ T.intercalate "\t" $ fst $ unzip $ testData 
  --T.putStrLn $ T.concat $ map float2color $ ((asValue y)::[Float])
  T.putStr "<html><head><title>test</title></head><body>"
  T.putStr $ T.concat $ map (\(t,c) -> T.concat ["<span style=background-color:", c, ">", t, "</span>"]) $ zip (fst $ unzip $ testData) (map float2color $ ((asValue y)::[Float]))
  T.putStrLn "</body></html>"

  
-- | returns (ys', ys, batchLoss) i.e. (predictions, groundtruths, batchloss)
feedForward :: Params -> (T.Text -> [Float]) -> (Tensor -> Tensor) -> [(Dat,Float)] -> (Tensor,Tensor,Tensor)
feedForward model oneHotFor toDevice dataSet = 
  let lstm = bilstmLayer (lstmParams model) (toDependent $ c0 model) (toDependent $ h0 model)
      embLayer = map (\w -> (toDependent $ w_emb model) `matmul` (toDevice $ asTensor $ oneHotFor w)) $ fst $ unzip $ dataSet
      y' = squeezeAll . cat (Dim 0) $ map (reshape [1,1] . sigmoid . linearLayer (mlpParams model)) $ fst $ unzip $ lstm embLayer
      y  = toDevice $ asTensor $ snd $ unzip $ dataSet
  in (y', y, binaryCrossEntropyLoss' y y')

-- | http://techarchforse.blogspot.com/2014/02/css-css-color-table.html
float2color :: Float -> T.Text
float2color float
  | 0.9 <= float                = "#FF00FF"
  | 0.85 <= float && float < 0.9 = "#FF22FF"
  | 0.8 <= float && float < 0.85 = "#FF44FF"
  | 0.75 <= float && float < 0.8 = "#FF55FF"
  | 0.7 <= float && float < 0.75 = "#FF77FF"
  | 0.65 <= float && float < 0.7 = "#FF99FF"
  | 0.6 <= float && float < 0.65 = "#FFBBFF"
  | 0.4 <= float && float < 0.6 = "#FFDDFF"
  | otherwise                   = "#FFFFFF"

