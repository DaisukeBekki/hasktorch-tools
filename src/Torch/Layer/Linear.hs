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
import Torch.Tensor          (Tensor(..),toCPU)
import Torch.Functional      (matmul,squeezeAll)
import Torch.Device          (Device(..))
import Torch.NN              (Parameter,Parameterized,Randomizable,sample)
import Torch.Autograd        (IndependentTensor(..),makeIndependent)
import Torch.Tensor.TensorFactories (randnIO')
import Torch.Tensor.Initializers    (xavierUniform')

data LinearHypParams = LinearHypParams {
  dev :: Device,
  inputDim :: Int,
  outputDim :: Int
  } deriving (Show, Eq)

data LinearParams = LinearParams { 
  weight :: Parameter,
  bias :: Parameter
  } deriving (Generic)

instance Parameterized LinearParams -- Generic

instance Randomizable LinearHypParams LinearParams where
  sample LinearHypParams{..} = do
    w <- makeIndependent =<< xavierUniform' dev [outputDim, inputDim]
    b <- makeIndependent =<< randnIO' dev [outputDim]
    return $ LinearParams w b

instance Show LinearParams where
  show LinearParams{..} = 
    "Parameters:\n"
    ++ (show $ toCPU $ toDependent weight)
    ++ "\nBias:\n"
    ++ (show $ toCPU $ toDependent bias) 
    
linearLayer :: LinearParams -> Tensor -> Tensor
linearLayer LinearParams{..} input =
  squeezeAll $ ((toDependent weight) `matmul` input) + (toDependent bias)

