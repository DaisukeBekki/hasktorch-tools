{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

--hasktorch
import Torch.Tensor (TensorLike(..),toCPU)
import Torch.Functional (mseLoss,add)
import Torch.NN         (sample)
import Torch.Optim      (GD(..))
import Torch.Train      (update,showLoss,zeroTensor,saveParams) --, loadParams)
import Torch.Control    (mapAccumM,foldLoop)
import Torch.Layer.Linear (LinearHypParams(..),linearLayer)
import Torch.Util.Chart (drawLearningCurve)

trainingData :: [([Float],Float)]
trainingData = [([1],2),([2],4),([3],6),([1],2),([3],7)]

testData :: [([Float],[Float])]
testData = [([3],[7])]

main :: IO()
main = do
  let iter = 150::Int
  initModel <- sample $ LinearHypParams 1 1
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let batchLoss = foldLoop trainingData zeroTensor $ \(input,output) loss ->
                      let y' = linearLayer model $ toCPU $ asTensor input
                          y = toCPU $ asTensor output
                      in add loss $ mseLoss y' y
        lossValue = (asValue batchLoss)::Float
    showLoss 5 epoc lossValue
    u <- update model opt batchLoss 5e-4
    return (u, lossValue)
  saveParams trainedModel "regression.model"
  --mapM_ (putStr . printf "%2.3f ") $ reverse allLosses
  drawLearningCurve "graph-reg.png" "Learning Curve" [("",reverse losses)]
  --loadedModel <- loadParams (Lin:@earHypParams 1 1) "regression.moxdel"
  --print loadedModel

