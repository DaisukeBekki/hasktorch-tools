module Torch.Util.Experiment (
  k_fold
  ) where

import qualified Data.Text    as T    --text
import qualified Data.Text.IO as T    --text
import qualified System.IO    as IO        --base
import qualified Data.List    as L       --base

-- | 与えられたsfile名のfileをK分割する関数
k_fold :: Int -> FilePath -> String -> IO()
k_fold k filepath filename = do
  content <- T.readFile $ filepath ++ "/" ++ filename ++ ".txt"
  --mapM_ (\j -> do
  let j = 0
      (_,train,test) = L.foldl' (\(i,train',test') l -> if mod i k == j
                                                            then ((i+1), train', l:test')
                                                            else ((i+1), l:train', test')     
                                ) (0,[],[]) $ T.lines content
  IO.withFile (filepath ++ "/" ++ filename ++ "_train.txt") IO.WriteMode (\h -> mapM_ (T.hPutStrLn h) train)
  IO.withFile (filepath ++ "/" ++ filename ++ "_test.txt") IO.WriteMode (\h -> mapM_ (T.hPutStrLn h) test)
  --  ) [0..k-1]
