{-# LANGUAGE OverloadedStrings, DeriveGeneric #-}

module Main where

--import Network.HTTP.Req (req, MonadHttp(..), GET(..), http, NoReqBody(..), jsonResponse, responseBody) --http-conduit
import GHC.Generics
import Control.Lens                           --lens
import Data.Aeson                             --aeson
import Data.Aeson.Lens 　                     --aeson-lens
import qualified Data.Text.Lazy as T          --text
import qualified Data.Text.Lazy.IO as T       --text
import qualified Data.Text.Lazy.Encoding as E --text
import qualified Data.ByteString.Lazy.Char8 as B    --bytestring

--getData :: (MonadHttp m) => m Value
--getData = responseBody <$> req
--  GET
-- - (http "https://github.com/tokyo-metropolitan-gov/covid19/blob/development/data/data.json")
--  NoReqBody
--  jsonResponse
--  mempty
  
main :: IO ()
main = do
--  let urlString = "https://github.com/tokyo-metropolitan-gov/covid19/blob/development/data/data.json"
--  response <- N.httpLBS $ N.parseRequest_ urlString
--  let json = B.toStrict $ N.getResponseBody response -- :: ByteString
  let json = "{\"num\":6,\"patients_summary\":[\"hey\"]}"
      Just dat = decode json :: Maybe Value
      Just p = dat ^? key "num"
  print p
