{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Prelude hiding (tanh) 
import Control.Monad (forM_)        --base
--import Data.List (cycle)          --base
--hasktorch
import Torch.Tensor       (TensorLike(..),toCPU)
import Torch.Functional   (mseLoss)
import Torch.NN           (sample)
import Torch.Train        (update,showLoss,sumTensors)
import Torch.Control      (mapAccumM)
import Torch.Optim        (GD(..))
import Torch.Layer.MLP    (MLPHypParams(..),ActName(..),mlpLayer)
import Torch.Util.Chart (drawLearningCurve)

trainingData :: [([Float],Float)]
trainingData = take 5 $ cycle [([1,1],0),([1,0],1),([0,1],1),([0,0],0)]

main :: IO()
main = do
  let iter = 1500::Int
  initModel <- sample $ MLPHypParams 2 2 1 Sigmoid Sigmoid
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let loss = sumTensors $ for trainingData $ \(input,output) ->
                 let y' = mlpLayer model $ toCPU $ asTensor input
                     y = toCPU $ asTensor output
                 in mseLoss y' y
        lossValue = (asValue loss)::Float
    showLoss 10 epoc lossValue 
    u <- update model opt loss 1e-1
    return (u, lossValue)
  drawLearningCurve "graph-xor.png" "Learning Curve" [("",reverse losses)]
  forM_ ([[1,1],[1,0],[0,1],[0,0]::[Float]]) $ \input -> do
    putStr $ show $ input
    putStr ": "
    putStrLn $ show ((mlpLayer trainedModel $ toCPU $ asTensor input))
  --print trainedParams
  where for = flip map


