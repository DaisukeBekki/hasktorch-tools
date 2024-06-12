module Torch.Train (
  update,
  showLoss,
  zeroTensor,
  sumTensors,
  saveParams,
  loadParams,
  saveState
  ) where

import Control.Monad (when)     --base
import Text.Printf   (printf)   --base

import Torch.Tensor    (Tensor(..),reshape, asTensor, asValue)
import Torch.TensorFactories (zeros')
import Torch.Functional (Dim(..),sumAll,cat)
import Torch.NN        (Parameterized(..),Randomizable(..),replaceParameters, Parameter)
import Torch.Autograd  (IndependentTensor(..),makeIndependent,grad)
import Torch.Optim     (Optimizer(..),Gradients(..), Adam(..))
import Torch.Serialize ( save, load, pickleSave, pickleLoad )
import Torch.Script (IValue(..))
import Torch.Autograd (IndependentTensor(..))

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

saveState :: Parameterized model => Int -> model -> Adam -> Tensor -> FilePath -> IO ()
saveState epoc model opt loss path = do
  let flattenParams = map toDependent $ flattenParameters model
  independentLoss <- makeIndependent loss
  let Adam {beta1 = b1, beta2 = b2, m1 = m1List, m2 = m2List, iter = iteration} = opt

  let lossEpoc = IVTensorList [asTensor (fromIntegral epoc :: Double) :: Tensor, toDependent independentLoss]
  let modelFlattened = IVTensorList flattenParams
  let betasIteration = IVTensorList [asTensor b1 :: Tensor, asTensor b2 :: Tensor, asTensor (fromIntegral iteration :: Double)]
  let m1 = IVTensorList m1List
  let m2 = IVTensorList m2List
  let state = IVGenericDict [(IVString "model", modelFlattened), (IVString "loss", lossEpoc), (IVString "betasIteration", betasIteration), (IVString "m1", m1), (IVString "m2", m2)]
  pickleSave state path
