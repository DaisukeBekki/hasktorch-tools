{-# LANGUAGE DeriveGeneric #-}

module Torch.Tensor.Initializers (
  kaimingUniform,
  kaimingUniform'
) where

import Prelude hiding (init)
import Torch.Tensor (Tensor(..))
import Torch.Device (Device(..))
import Torch.Functional (mulScalar,subScalar)
import Torch.Initializers (FanMode(..),NonLinearity(..),calculateFan,calculateGain,getter)
import Torch.Tensor.TensorFactories (randIO')

-- | Kaiming Initialization - Uniform
kaimingUniform :: Device -> FanMode -> NonLinearity -> [Int] -> IO Tensor
kaimingUniform dev mode nonlinearity shape = do
  init <- randIO' dev shape
  pure $ subScalar bound $ mulScalar (bound * 2.0) init
  where
    fanValue = fromIntegral $ getter mode (calculateFan shape)
    std = calculateGain nonlinearity / sqrt fanValue
    bound = sqrt 3.0 * std

kaimingUniform' :: Device -> [Int] -> IO Tensor
kaimingUniform' dev = kaimingUniform dev FanIn (LeakyRelu 0.0)


