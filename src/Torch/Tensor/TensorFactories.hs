{-# LANGUAGE DeriveGeneric #-}

module Torch.Tensor.TensorFactories (
  asTensor'',
  randIO',
  randnIO',
  randintIO',
  zeros',
  oneAtPos,
  oneAtPos2d
) where

import Torch.Tensor (Tensor(..),TensorLike(..),asTensor,asValue)
import Torch.TensorFactories (randIO,randnIO,randintIO,zeros)
import Torch.TensorOptions (defaultOpts,withDevice)
import Torch.Device (Device(..))

asTensor'' :: (TensorLike a) => Device -> a -> Tensor
asTensor'' dev v = asTensor' v $ withDevice dev defaultOpts

randIO' :: Device -> [Int] -> IO Tensor
randIO' dev shape = randIO shape $ withDevice dev defaultOpts

randnIO' :: Device -> [Int] -> IO Tensor
randnIO' dev shape = randnIO shape $ withDevice dev defaultOpts

randintIO' :: Device -> Int -> Int -> [Int] -> IO Tensor
randintIO' dev low high shape = randintIO low high shape $ withDevice dev defaultOpts

zeros' :: Device -> [Int] -> Tensor
zeros' dev shape = zeros shape $ withDevice dev defaultOpts

-- | Return a 1d Tensor with a 1 at the x position and 0 everywhere else
-- * x : The index of the 1
-- * size : Length of the wanted tensor
oneAtPos :: Int -> Int -> Tensor
oneAtPos x size = asTensor $ (replicate x (0 :: Float) ) ++ [1.0] ++ (replicate (size - x - 1) (0::Float) )  

-- | Return a 2d Tensor with a 1 at the x y position and 0 everywhere else
-- * x : The x index of the 1
-- * y : The y index of the 1
-- * size : Length of the wanted tensor
oneAtPos2d :: Int -> Int -> Int -> Tensor
oneAtPos2d x y size = asTensor $ (replicate x zer) ++ [oneLine] ++ (replicate (size-x-1) zer)
    where oneLine = asValue (oneAtPos y size) :: [Float]
          zer = replicate size (0 :: Float)
