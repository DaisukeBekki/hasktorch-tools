module Torch.Tensor.Util (
    unstack,
    oneHot',
    indexOfMax
) where

import Data.List (maximumBy)
import Torch.Tensor (Tensor(..),select,size,asTensor,asValue)
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


-- | Given a Tensor like [0.2, 0.3, 0.5], it will put the highest value to 1 and the others to 0
-- so it would give [0.0, 0.0, 1.0]
oneHot' :: Tensor -> Tensor
oneHot' t = asTensor $ replicate imax (0.0 :: Float) ++ [1.0] ++ replicate (tLen - imax - 1) (0.0 :: Float)
    where imax = indexOfMax (asValue t :: [Float])
          tLen = length (asValue t :: [Float])

-- | Return the index of the maximum value in the list
indexOfMax :: Ord a => [a] -> Int
indexOfMax xs = snd $ maximumBy (\x y -> compare (fst x) (fst y)) (zip xs [0..])
