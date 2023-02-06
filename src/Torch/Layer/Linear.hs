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
import Torch.Tensor.TensorFactories (randintIO')
import Torch.Tensor.Initializers    (xavierUniform')

data LinearHypParams = LinearHypParams {
  dev :: Device,
  ifBias :: Bool,
  inputDim :: Int,
  outputDim :: Int
  } deriving (Eq, Show)

data LinearParams = LinearParams { 
  weight :: Parameter,
  bias :: Maybe Parameter
  } deriving (Generic)

instance Parameterized LinearParams -- Generic

instance Randomizable LinearHypParams LinearParams where
  sample LinearHypParams{..} = 
    LinearParams
      <$> (makeIndependent =<< xavierUniform' dev [outputDim, inputDim])
      <*> if ifBias
            then (Just <$> (makeIndependent =<< randintIO' dev (-1) 1 [outputDim]))
            else return Nothing

instance Show LinearParams where
  show LinearParams{..} = 
    "Parameters:\n"
    ++ (show $ toCPU $ toDependent weight)
    ++ case bias of
         Just b -> "\nBias:\n"
                   ++ (show $ toCPU $ toDependent b)
         Nothing -> "" 
    
linearLayer :: LinearParams -> Tensor -> Tensor
linearLayer LinearParams{..} input =
  case bias of
    Just b -> squeezeAll $ ((toDependent weight) `matmul` input) + (toDependent b)
    Nothing -> squeezeAll $ ((toDependent weight) `matmul` input)

