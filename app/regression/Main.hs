module Main where

--hasktorch
import Torch.Tensor (asValue)
import Torch.Functional (mseLoss,add)
import Torch.Device     (Device(..),DeviceType(..))
import Torch.NN         (sample)
import Torch.Optim      (GD(..))
--hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Train      (update,showLoss,zeroTensor,saveParams,loadParams)
import Torch.Control    (mapAccumM,foldLoop)
import Torch.Layer.Linear (LinearHypParams(..),linearLayer)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools

trainingData :: [([Float],Float)]
trainingData = [([1],2),([2],4),([3],6),([1],2),([3],7)]

testData :: [([Float],Float)]
testData = [([3],7)]

main :: IO()
main = do
  let iter = 500::Int
      device = Device CPU 0
  initModel <- sample $ LinearHypParams device 1 1
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let batchLoss = foldLoop trainingData zeroTensor $ \(input,output) loss ->
                      let y' = linearLayer model $ asTensor'' device input
                          y = asTensor'' device output
                      in add loss $ mseLoss y y'
        lossValue = (asValue batchLoss)::Float
    showLoss 5 epoc lossValue
    u <- update model opt batchLoss 5e-4
    return (u, lossValue)
  saveParams trainedModel "regression.model"
  --mapM_ (putStr . printf "%2.3f ") $ reverse allLosses
  drawLearningCurve "graph-reg.png" "Learning Curve" [("",reverse losses)]
  loadedModel <- loadParams (LinearHypParams device 1 1) "regression.model"
  print loadedModel
  --
  let output = linearLayer loadedModel $ asTensor'' device $ fst $ head testData
      y' = (asValue output)::Float
      y = snd $ head testData
  putStr "\nPrediction: "
  putStrLn $ show y'
  putStr "Ground truth: "
  putStrLn $ show y
  putStr "Mse: "
  putStrLn $ show $ (y' - y) * (y' - y)
