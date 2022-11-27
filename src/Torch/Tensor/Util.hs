module Torch.Tensor.Util (
    unstack
) where

import Torch.Tensor (Tensor(..),select,size)
import Torch.Device (Device(..),DeviceType(..))
import Torch.Functional (Dim(..),stack)
import Torch.Tensor.TensorFactories (asTensor'')

-- | Transforming an input tensor of shape (n:N) to the list of Tensor of the chape N
-- | To stack the unstacked tensors, use stack (Dim 0).
unstack :: Tensor -- ^ input tensor
  -> [Tensor]
unstack t = for [0..(size 0 t - 1)] $ \j -> select 0 j t
  where for = flip map

-- pack :: [Tensor] -> Tensor
-- pack tensors = 

-- | test code
main :: IO() 
main = do
  let dev = Device CUDA 0
      u = asTensor'' dev [[1,2,3],[4,5,6]::[Float]]
      v = asTensor'' dev [[1,2],[3,4],[5,6]::[Float]]
      w = asTensor'' dev ([]::[Float])
  print $ unstack u
  print $ unstack v
  print $ unstack w
  print $ stack (Dim 0) $ unstack v
