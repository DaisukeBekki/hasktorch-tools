{-# OPTIONS -Wall #-}
{-# LANGUAGE OverloadedStrings #-}
--{-# LANGUAGE DeriveGeneric #-}
--{-# LANGUAGE MultiParamTypeClasses #-}
--{-# LANGUAGE RecordWildCards #-}

import qualified Data.Text           as T --text
import qualified Data.Text.IO        as T --text
import qualified Data.Char           as C --base
import qualified Data.List           as L --base
import qualified Data.Map.Strict     as M --containers
--hasktorch
import Torch.Tensor (TensorLike(..),toCPU)
import Torch.Functional (mseLoss,add)
import Torch.NN         (sample)
import Torch.Optim      (GD(..))
--hasktorch-tools
import Torch.Train      (update,showLoss,zeroTensor,saveParams) --, loadParams)
import Torch.Control    (mapAccumM,foldLoop)
import Torch.Layer.Linear (LinearHypParams(..),linearLayer)
import Torch.Util.Chart (drawLearningCurve)

main :: IO()
main = do
  -- | Preparation
  review_csv <- T.readFile "fire.csv"
  let review_csv' = T.replace "###" "\"\n" $ T.replace "\n" "" $ T.replace "\n\"\n" "###" $ T.replace ", 2020" " 2020" $ T.unlines $ tail $ T.lines review_csv
      review_data = map ((\x -> (T.words $ T.toLower $ T.filter (\c -> not (c `elem` [',','.','!','?'])) $ T.dropAround (\c -> C.isSpace c || c == '\"') (x!!5), (read $ T.unpack $ T.take 3 (x!!2))::Float)) . T.split (==',')) $ T.lines review_csv'
  -- | Create a dictionary
      document = concat $ fst $ unzip review_data
      document' = zip document ([1..]::[Int])
      wrds = fst $ unzip $ reverse $ L.sortOn snd $ M.toList $ M.fromListWith (+) document'
      dict = M.fromList $ zip wrds ([1..]::[Int])
  -- | Create training Data
      input_dim = length wrds
      trainingData = map (\(wds,score) -> (text2vec dict input_dim wds, score)) review_data
  -- | regression
      iter = 100::Int
  initModel <- sample $ LinearHypParams (input_dim + 1) 1
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let batchLoss = foldLoop trainingData zeroTensor $ \(input,output) loss ->
                      let y' = linearLayer model $ toCPU $ asTensor input
                          y = toCPU $ asTensor output
                      in add loss $ mseLoss y' y
        lossValue = (asValue batchLoss)::Float
    showLoss 5 epoc lossValue
    u <- update model opt batchLoss 5e-4
    return (u, lossValue)
  drawLearningCurve "graph-amazon.png" "Learning Curve" [("",reverse losses)]
  --saveParams trainedModel "amazon_review.model"
  --loadedModel <- loadParams (LinearHypParams input_dim 1) "regression.moxdel"

-- | Some functions for dictionary

zeroVecOfLength :: Int -> [Float]
zeroVecOfLength i | i <= 0 = []
                  | otherwise = (0::Float):(zeroVecOfLength (i-1))

text2vec :: (M.Map T.Text Int) -> Int -> [T.Text] -> [Float]
text2vec dict dim wrds = L.foldl' (\vec w -> case addNth vec dim (word2order dict w) of
                                               Just v -> v
                                               Nothing -> vec
                                  ) (zeroVecOfLength (dim+1)) wrds 

word2order :: (M.Map T.Text Int) -> T.Text -> Int
word2order dict wd = case M.lookup wd dict of
                       Just i -> i
                       Nothing -> 0

addNth :: [Float] -> Int -> Int -> Maybe [Float]
addNth vec dim i
  | i < 0 = Nothing
  | i >= dim = Nothing
  | otherwise = addNth' vec i

addNth' :: [Float] -> Int -> Maybe [Float]
addNth' [] i = Nothing
addNth' (h:t) i
  | i == 0 = Just $ (h + (1::Float)):t
  | otherwise = addNth' t (i-1)

