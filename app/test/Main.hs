{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-} -- これか
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE PatternSynonyms #-} -- これか
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE RankNTypes #-}

module Main where

import GHC.TypeNats
import Torch.Tensor        (Tensor(..),TensorLike(..),TensorIndex(..),toCPU,device,dtype,select,(@@))
import Torch.TensorOptions (defaultOpts, withDevice)
import Torch.Functional    (matmul,transpose)
import Torch.Device        (Device(..),DeviceType(..))
import Torch.DType 
--import Torch.Typed.Tensor


main :: IO()
main = do
--  let u = UnsafeMkTensor @'( 'D.CPU,0) @D.Float @[1,3] $ D.asTensor ([[1,2,3]]::[[Float]]) -- (optionsRuntimeShape @shape @dtype @device) (D.withDevice (optionsRuntimeDevice @shape @dtype @device) . D.withDType (optionsRuntimeDType @shape @dtype @device) $ D.defaultOpts)
--      v = UnsafeMkTensor @'( 'D.CPU,0) @D.Float @[3,1] $ D.asTensor ([[2],[3],[4]]::[[Float]])
  let u = toCPU $ asTensor [[1,2],[3,4],[4,5]::[Float]]
      v = toCPU $ asTensor [[1,2,3]::[Float]]
  putStrLn $ "u = " ++ (show u)
  putStrLn $ "device u = " ++ (show $ device u)
  putStrLn $ "dtype u = " ++ (show $ dtype u)
  putStrLn $ show $ u@@((1,0)::(Int,Int))
  putStrLn $ "select u 0 1 = " ++ (show $ select u 1 0)
  putStrLn $ "v = " ++ (show v)
  putStrLn $ "tv = " ++ (show $ transpose v 1 0)
  putStrLn $ "tu(tv) = " ++ (show $ matmul (transpose u 1 0) $ transpose v 1 0)
  --putStrLn $ "vu = " ++ (show $ matmul v u)

--mkTensor :: (TensorLike a) => a -> Tensor
--mkTensor = flip asTensor' (withDevice (Device CPU 0) defaultOpts)
