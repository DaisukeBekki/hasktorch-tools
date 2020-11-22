{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Data.Text as T        --text
import qualified Data.Text.IO as T     --text
import Control.Monad (forM_,unless)   --base
import qualified System.Directory as D --directory
import System.FilePath.Posix ((</>),isExtensionOf) --filepath
import qualified Text.Juman as J       --juman-tools

dataFolder :: FilePath
dataFolder = "/home/bekki/dropbox/Public/MyData/nec2020/brat_data_20201102_rev/単語区切り修正後"

main :: IO()
main = do
  D.doesDirectoryExist dataFolder
    >>= (flip unless $ ioError $ userError $ dataFolder ++ " not exist")
  files <- filter (isExtensionOf "txt") <$> D.listDirectory dataFolder
  forM_ files $ \file -> do
    J.fromFile $ dataFolder </> file
    >>= mapM_ (J.printJumanData . J.jumanParser)

      
