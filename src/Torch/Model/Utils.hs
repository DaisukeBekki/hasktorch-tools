module Torch.Model.Utils (
  confusionMatrix,
  confusionMatrixPlot
  ) where

import Torch.Functional (add)
import Torch.Tensor (Tensor(..), asTensor, asValue)
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
