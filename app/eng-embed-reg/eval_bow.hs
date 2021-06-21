{-# LANGUAGE OverloadedStrings, ExtendedDefaultRules, DeriveGeneric, RecordWildCards, MultiParamTypeClasses #-}

module Main where

import Prelude hiding (tanh)
import Control.Monad (when,filterM,forM,forM_)      --base
import Control.Exception (throw)        --base
import System.FilePath ((</>),isExtensionOf)           --filepath
import System.Directory (doesDirectoryExist, doesFileExist, listDirectory) --directory
import qualified Data.List as L              --base
import qualified Data.Text.Lazy as T         --text
import qualified Data.Text.Lazy.IO as T      --text
import qualified Data.ByteString.Lazy as BL  --bytestring
import qualified Data.JsonStream.Parser as J --json-stream
import qualified GHC.Generics as G          --base
import qualified Data.Aeson            as A --aeson
import qualified Data.ByteString.Char8 as BC --bytestring 
import qualified Data.Yaml             as Y --yaml
-- | hasktorch
import Torch.Tensor (asValue)
import Torch.Functional (Dim(..),KeepDim(..),stack,sumDim,squeezeAll,mseLoss,add,divScalar,mulScalar,sumAll)
import Torch.Device (Device(..),DeviceType(..))
import Torch.DType (DType(..))
import Torch.NN (Parameterized(..),Randomizable(..),sample)
import Torch.Autograd (IndependentTensor(..))
import Torch.Optim (GD(..))
-- | hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'',randnIO')
import Torch.Train (update,showLoss,zeroTensor,saveParams,loadParams,sumTensors)
import Torch.Control (mapAccumM,trainLoop,makeBatch)
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.MLP (MLPHypParams(..),MLPParams(..),ActName(..),mlpLayer)
import Torch.Util.Directory (checkDir,checkFile)
import Torch.Util.Dict (sortWords, oneHotFactory)
import Torch.Util.Chart (drawLearningCurve)
import Torch.Config.EngEmbedRegConfig (Config(..),getConfig)

main :: IO ()
main = do
  (_,config) <- getConfig -- | Reading config.yaml
  let debug = False
      device = Device CPU 0 -- | use CPU
      dicFile = dic_filepath config -- | FilePath of a save file
  (DictionaryData dict) <- loadDictionary dicFile -- | Loading the lexicon
  let (oneHot, dicDim) = oneHotFactory dict -- | Building a one-hot vector factory from the lexicon
      hypParams = MLPHypParams device dicDim [(300,Selu),(1,Sigmoid)]
  model <- loadParams hypParams "amazon_embedding.model"
  comment <- T.readFile "comment.txt"
  let numWrds = (fromIntegral $ length $ T.words comment)::Float
      prediction = mulScalar (5::Float) $ mlpLayer model $ divScalar numWrds $ squeezeAll $ sumDim (Dim 0) KeepDim Float $ stack (Dim 0) $ map (asTensor'' device . oneHot) $ T.words comment
  putStr "Review: "
  T.putStrLn comment
  putStr "Predicted score: "
  putStrLn $ show ((asValue prediction)::Float)

type AmazonReview = (Float, [T.Text])
data DictionaryData = DictionaryData [T.Text] deriving (Show, G.Generic)

instance A.FromJSON DictionaryData
instance A.ToJSON DictionaryData

loadDictionary :: FilePath -> IO(DictionaryData)
loadDictionary filepath = do
  whenM (not <$> doesFileExist filepath) $
    throw (userError $ filepath ++ " does not exist.")
  content <- BC.readFile filepath
  let parsedDic = Y.decodeEither' content :: Either Y.ParseException DictionaryData
  case parsedDic of
    Left parse_exception -> error $ "Could not parse dic file " ++ filepath ++ ": " ++ (show parse_exception)
    Right dic -> return dic
  where whenM s r = s >>= flip when r
