module Torch.Model.Utils (
    confusionMatrix,
    confusionMatrixPlot,
    accuracy,
    precision,
    recall,
    f1,
    macroAvg,
    weightedAvg
  ) where

import Torch.Functional (add, mul, div, sumAll)
import Torch.Tensor (Tensor(..), asTensor, asValue)
import Torch.TensorFactories (zeros')
import Torch.Tensor.TensorFactories (oneAtPos2d)
import Torch.Tensor.Util (indexOfMax,oneHot')
import Torch.Layer.MLP      (MLPParams(..))

import Graphics.Matplotlib (Matplotlib(..), o2, title, xlabel, ylabel, colorbar, mp, setSizeInches, pcolor, text, (%), (@@), (#))

-- | Return the confusion matrix of a model
-- * model : The model
-- * forward : The forward function for the model 
-- * dataSet : The dataset
confusionMatrix :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> [[Int]]
confusionMatrix model forward dataSet = map (map round) floatRes
    where guesses = map (\(input, _) -> oneHot' $ forward model input) dataSet
          expected = map snd dataSet
          (_, output) = dataSet !! 0
          classificationSize = length $ (asValue output :: [Float])
          floatRes = asValue (foldr1 (Torch.Functional.add) [ oneAtPos2d (indexOfMax (asValue expect :: [Float])) (indexOfMax (asValue guess :: [Float])) classificationSize | (expect, guess) <- zip expected guesses]) :: [[Float]]

-- | Return a Matplotlib of the confusion matrix given in arg
-- * m : confusion matrix
-- * labels : the labels of the classes 
confusionMatrixPlot :: [[Int]] -> [String] -> Matplotlib
confusionMatrixPlot m labels = 
    let valuesText = concat $ [[text ((x+0.5) :: Double) ((y+0.5) :: Double) (show v) @@ [o2 ("ha" :: String) ("center" :: String), o2 ("va" :: String) ("center" :: String), o2 ("fontsize" :: String) (8 :: Int)] | (v,x) <- zip l [0..]] | (l,y) <- zip m [0..]]
        labelsText = [
            text (-0.1 :: Double) ((y+0.5) :: Double) label @@ [o2 ("ha" :: String) ("right" :: String), o2 ("va" :: String) ("center" :: String), o2 ("fontsize" :: String) (8 :: Int)] 
            % text ((y + 0.5) :: Double) (((fromIntegral $ length m) + 0.1) :: Double) label @@ [o2 ("ha" :: String) ("center" :: String), o2 ("va" :: String) ("top" :: String), o2 ("fontsize" :: String) (8 :: Int), o2 ("rotation" :: String) (90 :: Int)] 
            | (label,y) <- zip labels [0..]]
        numRows = length m
        newSizeX = round $ fromIntegral numRows / 2.0 * 1.3
        newSizeY = round $ fromIntegral numRows / 2.0
    in pcolor m @@ [o2 ("edgecolors" :: String) ("k" :: String), o2 ("linewidth" :: String) (1 :: Int)]
       % foldr (%) ( foldr (%) (
        title "Confusion Matrix"
        % setSizeInches (newSizeX :: Int) (newSizeY :: Int)
        % xlabel "Actual" 
        % ylabel "Expected" 
        % colorbar 
        % mp # ("ax.invert_yaxis()" :: String)) labelsText) valuesText

-- | Return the accuracy of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
accuracy :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
accuracy model forward dataSet = (sum results) / (fromIntegral (length results))
    where results = map (\(input, output) -> if (indexOfMax $ (asValue (forward model input) :: [Float])) == (indexOfMax $ (asValue output :: [Float])) then 1 else 0) dataSet
       
-- | Return the precision of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
precision :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Tensor
precision model forward dataSet = Torch.Functional.div tp (Torch.Functional.add tp fp)
    where (tp, _, fp) = getTpFnFp model forward dataSet

-- | Return the recall of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
recall :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Tensor
recall model forward dataSet = Torch.Functional.div tp (Torch.Functional.add tp fn)
    where (tp, fn, _) = getTpFnFp model forward dataSet

-- | Return the f1 score of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
f1 :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Tensor
f1 model forward dataSet = Torch.Functional.div tp (Torch.Functional.add tp (Torch.Functional.div (Torch.Functional.add fn fp) 2.0))
    where (tp, fn, fp) = getTpFnFp model forward dataSet

-- | Return the macro average of the f1 score of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
macroAvg :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
macroAvg model forward dataSet = (asValue (sumAll f1score) :: Float) / (fromIntegral $ length (asValue f1score :: [Float]))
    where f1score = f1 model forward dataSet

-- | Return the weighted average of the f1 score of a model
-- * model : The model
-- * forward : The forward function for the model
-- * dataSet : The dataset
weightedAvg :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
weightedAvg model forward dataSet = result
    where expecteds = map snd dataSet
          weights = Torch.Functional.div (foldl1 (Torch.Functional.add) expecteds) (fromIntegral $ length dataSet) 
          f1score = f1 model forward dataSet
          weightedF1 = Torch.Functional.mul weights f1score
          result = asValue (sumAll weightedF1) :: Float


----

-- | Return the true positive, false negative and false positive of one class
getOneTpFnFp :: Tensor -> Tensor -> (Tensor,Tensor,Tensor)
getOneTpFnFp expected guess = result
    where expectedValue = asValue expected :: [Float]
          guessValue = asValue guess :: [Float] 
          nul = zeros' [(length (asValue expected :: [Float]))]
          result = if indexOfMax expectedValue == indexOfMax guessValue then (expected, nul, nul) else (nul, expected, guess)

-- | Return the true positive, false negative and false positive of a classification model
getTpFnFp :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> (Tensor,Tensor,Tensor)
getTpFnFp model forward dataSet = (tp, fn, fp)
    where fTpFnFp = [ getOneTpFnFp output $ oneHot' $ forward model input | (input, output) <- dataSet]
          tp = foldl1 (Torch.Functional.add) [tps | (tps, _, _) <- fTpFnFp]
          fn = foldl1 (Torch.Functional.add) [fns | (_, fns, _) <- fTpFnFp]
          fp = foldl1 (Torch.Functional.add) [fps | (_, _, fps) <- fTpFnFp]
