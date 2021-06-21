{-# LANGUAGE DeriveGeneric #-}

module Torch.Tensor.Initializers (
  kaimingUniform,
  kaimingUniform',
  kaimingNormal,
  kaimingNormal',
  xavierUniform,
  xavierUniform',
  xavierNormal,
  xavierNormal'
) where

import Prelude hiding (init)
import Torch.Tensor (Tensor(..))
import Torch.Device (Device(..))
import Torch.Functional (mulScalar,subScalar)
import Torch.Initializers (FanMode(..),NonLinearity(..),calculateFan,calculateGain,getter)
import Torch.Tensor.TensorFactories (randIO',randnIO')

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

-- | Kaiming Initialization - Normal
kaimingNormal :: Device -> FanMode -> NonLinearity -> [Int] -> IO Tensor
kaimingNormal dev mode nonlinearity shape = mulScalar std <$> randnIO' dev shape
  where
    fanValue = fromIntegral $ getter mode (calculateFan shape)
    std = calculateGain nonlinearity / sqrt fanValue

kaimingNormal' :: Device -> [Int] -> IO Tensor
kaimingNormal' dev = kaimingNormal dev FanIn (LeakyRelu 0.0)

-- | Xavier Initialization - Uniform
xavierUniform :: Device -> Float -> [Int] -> IO Tensor
xavierUniform dev gain shape = do
  init <- randIO' dev shape
  pure $ subScalar bound $ mulScalar (bound * 2.0) init
  where
    (fanIn, fanOut) = calculateFan shape
    std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))
    bound = sqrt 3.0 * std

-- | Xavier Initialization - Normal
xavierNormal :: Device -> Float -> [Int] -> IO Tensor
xavierNormal dev gain shape = do
  init <- randnIO' dev shape
  pure $ mulScalar std init
  where
    (fanIn, fanOut) = calculateFan shape
    std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))

xavierUniform' :: Device -> [Int] -> IO Tensor
xavierUniform' dev = xavierUniform dev 1.0

xavierNormal' :: Device -> [Int] -> IO Tensor
xavierNormal' dev = xavierNormal dev 1.0

