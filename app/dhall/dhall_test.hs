{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}

import Data.Text
import Dhall

data Label = X | B | E | I deriving (Eq,Show,Read,Bounded,Enum)

data Rakkyo = Rakkyo
  { char :: Text
  , labels1 :: Vector Text -- メモ：Labelだとdhallでは読めない。あとでreadでLabelに変換する。
  , vals1 :: Vector Double -- メモ：Floatだとdhallでは読めない。あとでFloatにキャストする。
  , labels2 :: Vector Text
  , vals2 :: Vector Double
  , labels3 :: Vector Text
  , vals3 :: Vector Double 
  , labels4 :: Vector Text
  , vals4 :: Vector Double
  , labels5 :: Vector Text
  , vals5 :: Vector Double
  } deriving (Generic, Show)

instance Interpret Rakkyo

main :: IO ()
main = do
  x <- input auto "./data.dhall"
  print (x :: [Rakkyo])
  -- 以下は参考
  print ((read "X") :: Label)
