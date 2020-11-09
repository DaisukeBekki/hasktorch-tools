module Torch.Train (
  update,
  showLoss,
  zeroTensor,
  sumTensors,
  saveParams,
  loadParams
  ) where

import Control.Monad (when)     --base
import Text.Printf   (printf)   --base

import Torch.Tensor    (Tensor(..),reshape)
import Torch.TensorFactories (zeros')
import Torch.Functional (Dim(..),sumAll,cat)
import Torch.NN        (Parameterized(..),Randomizable(..),replaceParameters)
import Torch.Autograd  (IndependentTensor(..),makeIndependent,grad)
import Torch.Optim     (Optimizer(..),Gradients(..))
import Torch.Serialize (save,load)

type Loss = Tensor
type LearningRate = Tensor

update :: (Parameterized model, Optimizer opt) =>
  model -> opt -> Loss -> LearningRate -> IO (model, opt)
update model opt lossTensor learningRate = do
  let params = flattenParameters model
      gradients = Gradients $ grad lossTensor params
      (params', opt') = step learningRate gradients (fmap toDependent params) opt
  newParams <- mapM makeIndependent params'
  return (replaceParameters model newParams, opt')

showLoss :: Int -> Int -> Float -> IO()
showLoss modulus epoc lossValue = 
  when (epoc `mod` modulus == 0) $ 
    putStrLn $ "Epoc: " ++ printf "%4d" epoc ++ "  --->  Loss: " ++ printf "%.6f" lossValue

zeroTensor :: Tensor
zeroTensor = zeros' []

sumTensors :: [Tensor] -> Tensor
sumTensors ts = sumAll $ cat (Dim 0) $ map (reshape [1]) ts

saveParams :: (Parameterized param) => param -> FilePath -> IO()
saveParams params = save (map toDependent $ flattenParameters params) 

loadParams :: (Parameterized param, Randomizable hyp param) => hyp -> FilePath -> IO(param)
loadParams hyps filepath = do
  tensors <- load filepath
  loaded_params <- mapM makeIndependent tensors
  model <- sample hyps
  return $ replaceParameters model loaded_params

