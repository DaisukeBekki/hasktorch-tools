module Torch.Util.Chart (
  drawLearningCurve
  ) where

--import Data.List (transpose)    --base
import Graphics.Gnuplot.Simple  --gnuplot

type Name = String
type Loss = Float
type LearningChart = (Name,[Loss]) --Name of an experiment and list of loss values

-- | LearningChart
-- |   Create a PNG file of learning charts
drawLearningCurve ::
  FilePath           -- ^ The filepath for the PNG file
  -> String          -- ^ The caption of the PNG file
  -> [LearningChart] -- ^ The list of data
  -> IO()
drawLearningCurve filepath title learningcharts = do
  let maxepoch   = fromIntegral $ maximum $ for learningcharts (length . snd)
      styleddata = for learningcharts formatLearningChart -- | [(PlotStyle, [(Double,Double)])]
      graphstyle = [(PNG filepath),(Title title),(XLabel "Epoch"),(YLabel "Loss"),(XRange (0,maxepoch::Double))]
  plotPathsStyle graphstyle styleddata
  where formatLearningChart (name,losses) =
          let name' = PlotStyle LinesPoints (CustomStyle [LineTitle name])
              losses' = for losses $ \loss -> (realToFrac loss)::Double
              epochs' = for [(0::Int)..] $ \i -> (fromIntegral i)::Double
          in (name', zip epochs' losses')
        for :: [a] -> (a -> b) -> [b]
        for = flip map
