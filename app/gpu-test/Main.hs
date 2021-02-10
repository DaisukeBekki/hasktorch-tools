module Main where

import Torch
--import Torch.Tensor.TensorFactories (asTensor'')

main :: IO ()
main = do
  let dev = (Device CUDA 0)
  v <- randIO [3] $ withDevice dev defaultOpts
  putStrLn $ show $ device v
  putStrLn $ show v
  let w = asTensor' ([1,2,3]::[Float]) $ withDevice (Device CPU 0) defaultOpts
  putStrLn $ show $ device w
  putStrLn $ show w 
  let u = asTensor' ([1,2,3]::[Float]) $ withDevice dev defaultOpts
      u' = toCPU u
  putStrLn $ show $ device u'
  putStrLn $ show u'
  putStrLn $ show $ device u 
  putStrLn $ show u
  --print $ u - v
  --print $ (u - v)@@(0::Int)
  --print ((asValue u)::DType)
