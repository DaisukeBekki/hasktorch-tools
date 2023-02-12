module Main where

--hasktorch
import Torch.Tensor (TensorLike(..),asValue,shape,dim,sliceDim,reshape)
import Torch.Functional (Dim(..),add,transpose,matmul,softmax,maskedSelect)
import Torch.Device     (Device(..),DeviceType(..))
--hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'',randnIO')
import Torch.NN           (sample)
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
--import Torch.Layer.ProtoType.Transformer (sdpAttention)

main :: IO()
main = do
  let dev = Device CUDA 0
  {-
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
  -}
  let v = asTensor'' dev [3,2,1::Float] -- shape [3]
  print v
  let k = asTensor'' dev [[3],[2],[1]::[Float]] -- shape [3,1]
  print k
  let q = asTensor'' dev [[[3],[2],[1]],[[5],[5],[5]]::[[Float]]] -- shape [2,3,1]
  print q
  let inputDim = 3
      outputDim = 4
  linearHypParams <- sample $ LinearHypParams dev False inputDim outputDim
  print $ linearLayer linearHypParams v
  print $ linearLayer linearHypParams k
  print $ linearLayer linearHypParams q
  --print $ matmul q k
  --print $ matmul k q
  --let v = asTensor'' dev [[1,0,2,2],[2,1,2,0],[2,2,0,1]::[Float]]
  --print $ sdpAttention dev 2 4 3 q k v
  --print $ transpose (Dim 0) (Dim 1) k
  --print $ matmul q (transpose (Dim 0) (Dim 1) k)
  --print $ softmax (Dim 0) $ asTensor'' dev [1,2,3::Float]
  --print $ softmax (Dim 1) $ reshape [1,3] $ asTensor'' dev [1,2,3::Float]
