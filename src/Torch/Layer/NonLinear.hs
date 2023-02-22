{-# LANGUAGE DeriveGeneric #-}

module Torch.Layer.NonLinear (
  ActName(..),
  decodeAct
  ) where

import Prelude hiding (tanh)
import GHC.Generics              --base
import qualified Data.Aeson as A --aeson
import Torch.Tensor (Tensor(..)) --hasktorch
import Torch.Functional   (sigmoid,relu,elu',selu) --hasktorch

data ActName = Id | Sigmoid | Tanh | Relu | Elu | Selu deriving (Eq,Show,Read,Generic)

instance A.FromJSON ActName
instance A.ToJSON ActName

decodeAct :: ActName -> Tensor -> Tensor
decodeAct actname = case actname of
                   Id  -> id
                   Sigmoid -> sigmoid
                   Tanh -> tanh
                   Relu -> relu
                   Elu -> elu'
                   Selu -> selu
  -- | HACK : Torch.Functional.tanhとexpが `Segmentation fault`になるため
  where tanh x = 2 * (sigmoid $ 2 * x) - 1