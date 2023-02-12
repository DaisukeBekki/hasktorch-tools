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

import GHC.Generics          --base
import Data.List             (singleton) --base
--hasktorch
import Torch.Tensor          (Tensor(..),toCPU)
import Torch.Functional      (matmul)
import Torch.Device          (Device(..))
import Torch.NN              (Parameter,Parameterized,Randomizable,sample)
import Torch.Autograd        (IndependentTensor(..),makeIndependent)
--hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'',randintIO')
--import Torch.Tensor.Initializers    (xavierUniform')

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
    let denom = asTensor'' dev $ singleton $ sqrt $ ((fromIntegral outputDim)::Float)
    m <- randintIO' dev (-1) 1 [outputDim, inputDim] 
    b <- randintIO' dev (-1) 1 [outputDim]
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
    
linearLayer :: LinearParams -- ^ model
  -> Tensor -- ^ input tensor of shape [batchSize, inputDim, 1]
  -> Tensor -- ^ output tensor of shape [batchSize, outputDim, 1]
linearLayer LinearParams{..} input =
  case bias of
    Just bias' -> ((toDependent weight) `matmul` input) + (toDependent bias')
    Nothing -> ((toDependent weight) `matmul` input)

