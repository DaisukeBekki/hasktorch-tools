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
import Torch.Tensor          (Tensor(..))
import Torch.TensorFactories (randnIO')
import Torch.Functional      (matmul,squeezeAll)
import Torch.NN              (Parameter,Parameterized,Randomizable,sample)
import Torch.Autograd        (IndependentTensor(..),makeIndependent)
import Torch.Initializers    (kaimingUniform')

data LinearHypParams = LinearHypParams {
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
    w <- makeIndependent =<< kaimingUniform' [outputDim, inputDim]
    b <- makeIndependent =<< randnIO' [outputDim]
    return $ LinearParams w b

instance Show LinearParams where
  show LinearParams{..} = 
    "Parameters:\n"
    ++ (show $ toDependent weight)
    ++ "\nBias:\n"
    ++ (show $ toDependent bias) 
    
linearLayer :: LinearParams -> Tensor -> Tensor
linearLayer LinearParams{..} input =
  squeezeAll $ ((toDependent weight) `matmul` input) + (toDependent bias)
--linearLayer LinearParams{..} = linear (Linear weight bias)

