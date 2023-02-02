module Main where

--hasktorch
import Torch.Tensor (asValue,shape,dim,sliceDim)
import Torch.Functional (add)
import Torch.Device     (Device(..),DeviceType(..))
--hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'',randnIO')

main :: IO()
main = do
  let dev = Device CUDA 0
      u = asTensor'' dev [[1,2,3],[2,3,4],[5,6,7],[8,9,10]::[Float]]
      m = asTensor'' dev [[2,3],[4,5]::[Float]]
  x <- randnIO' dev [1,3]
  y <- randnIO' dev [2,3]
  z <- randnIO' dev [3]
  --when (shape u /= [2,2]) $ ioError (userError $ "illegal shape: " ++ (show $ shape u))
  print x
  print y
  print $ dim x
  print $ dim y
  print $ dim z  
  print u
  --print $ sliceDim 0 1 3 0 u
  print $ head $ shape u
  print $ sliceDim 0 0 1 1 u
  print $ shape $ sliceDim 0 0 1 1 u
