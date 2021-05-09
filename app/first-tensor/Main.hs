module Main where

--hasktorch
import Torch.Tensor (asValue)
import Torch.Functional (add)
import Torch.Device     (Device(..),DeviceType(..))
--hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'')

main :: IO()
main = do
  let dev = Device CPU 0
  let u = asTensor'' dev [1,2,3::Int]
  print u

