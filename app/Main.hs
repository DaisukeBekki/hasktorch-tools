module Main where

import Torch

main :: IO ()
main = do
  let u = asTensor ([[1,2],[2,3],[4,5]]::[[Int]])
      v = asTensor ([[1,1],[1,1],[1,1]]::[[Int]])
  print $ u - v
  print $ (u - v)@@(0::Int)
--  print ((asValue u)::DType)
