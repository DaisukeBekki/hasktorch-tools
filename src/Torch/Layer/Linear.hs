{-# LANGUAGE DeriveGeneric #-}
-- {-# LANGUAGE TypeApplications #-}
-- {-# LANGUAGE ScopedTypeVariables #-}
-- {-# LANGUAGE MultiParamTypeClasses #-}
-- {-# LANGUAGE AllowAmbiguousTypes #-}

module Torch.Layer.Linear (
  LinearHypParams(..),
  LinearParams(..),
  linearLayer
  ) where  

import GHC.Generics                       --base
import Control.Monad    (unless)     --base
import System.IO.Unsafe (unsafePerformIO) --base
--hasktorch
import Torch.Tensor          (Tensor(..),toCPU,shape,reshape)
import Torch.Functional      (matmul)
import Torch.Device          (Device(..))
import Torch.NN              (Parameter,Parameterized,Randomizable,sample)
import Torch.Autograd        (IndependentTensor(..),makeIndependent)
--hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Tensor.Initializers    (xavierUniform')

data LinearHypParams = LinearHypParams {
  dev :: Device,
  hasBias :: Bool,
  inputDim :: Int,
  outputDim :: Int
  } deriving (Eq, Show)

data LinearParams = LinearParams { 
  weight :: Parameter,
  bias :: Maybe Parameter
  } deriving (Generic)

instance Parameterized LinearParams -- Generic

instance Randomizable LinearHypParams LinearParams where
  sample LinearHypParams{..} = do
    let denom = asTensor'' dev $ [sqrt $ ((fromIntegral outputDim)::Float)]
    m <- xavierUniform' dev [outputDim, inputDim] 
    b <- xavierUniform' dev [outputDim, 1]
    LinearParams
      <$> makeIndependent (m / denom) -- denomでnormalize
      <*> if hasBias
            then Just <$> (makeIndependent b)
            else return Nothing

instance Show LinearParams where
  show LinearParams{..} = 
    "Parameters:\n"
    ++ (show $ toCPU $ toDependent weight)
    ++ case bias of
         Just bias' -> "\nBias:\n" ++ (show $ toCPU $ toDependent bias')
         Nothing    -> "" 

debug :: Bool
debug = True

-- | Linear layer that supports broadcasting
linearLayer :: LinearParams -- ^ model
  -> Tensor -- ^ input tensor <..., inputDim>
  -> Tensor -- ^ output tensor <..., outputDim>
linearLayer LinearParams{..} input = unsafePerformIO $ do
  --when (debug) $ print $ shape $ input
  --when (debug) $ print $ shape $ toDependent weight
  let inputShape = shape input
      revshape@(inputDim:batchDims) = reverse inputShape
      matrix = toDependent weight
      matrixShape@(cols:(rows:_)) = shape matrix
  unless (inputDim == rows) $ print $ 
    "illformed input for linear layer: " ++ (show matrixShape) ++ " and " ++ (show inputShape)
    ++ " inputDim = " ++ (show inputDim) ++ " row = " ++ (show rows)
  let newInput = reshape (reverse (1:revshape)) input
      rawOutput = case bias of
                    Just bias' -> (matrix `matmul` newInput) + (toDependent bias')
                    Nothing -> (matrix `matmul` newInput)
  return $ reshape (reverse $ (cols:batchDims)) rawOutput

