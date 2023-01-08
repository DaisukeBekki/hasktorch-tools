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
import Torch.Tensor (asValue,reshape)
import Torch.Functional (Dim(..),KeepDim(..),stack,sumDim,squeezeAll,mseLoss,add,divScalar,mulScalar,sumAll,cat,sigmoid,tanh,nllLoss',logSoftmax,softmax)
import Torch.Device (Device(..),DeviceType(..))
import Torch.DType (DType(..))
import Torch.NN (Parameterized(..),Randomizable(..),Parameter,sample)
import Torch.Autograd (IndependentTensor(..),makeIndependent)
import Torch.Optim (GD(..))
-- | hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'',randnIO')
import Torch.Train (update,showLoss,zeroTensor,saveParams,loadParams,sumTensors)
import Torch.Control (mapAccumM,trainLoop,makeBatch)
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.MLP (MLPHypParams(..),MLPParams(..),ActName(..),mlpLayer)
import Torch.Layer.LSTM (LstmHypParams(..),LstmParams(..),lstmLayers)
import ML.Util.Directory (checkDir,checkFile)  --nlp-tools
import ML.Util.Dict (sortWords, oneHotFactory) --nlp-tools
import ML.Exp.Chart (drawLearningCurve)        --nlp-tools
import Torch.Config.EngEmbedRegConfig (Config(..),getConfig)

main :: IO ()
main = do
  -- | Preprocessing
  let debug = True
      device = Device CPU 0 -- | use CPU
  (_,config) <- getConfig -- | Reading config.yaml
  let --wlpdataDir = wipdata_dirpath config -- | FilePath of a directory of wlp files
      dicFile = dic_filepath config -- | FilePath of a save file
      amazonreviewFile = amazonreview_filepath config -- | TSV file containing amazon reviews for Fire Tablet
  checkFile amazonreviewFile
  amazonReviewsTSV <- T.readFile amazonreviewFile -- | Reading amazon reviews
  let amazonReviews = map (\lin -> let items = T.splitOn "\t" lin -- | Separating each line by the tab
                                       rating = (ceiling ((read (T.unpack $ items!!7))::Double))::Int -- | Read off the rating
                                       comment = T.words $ T.replace "," "" $ T.replace "." "" $ items!!1
                                   in (rating,comment)) $ tail $ T.lines amazonReviewsTSV -- | Returns a list of pairs
      wrds = drop 50 $ sortWords 3 $ concat $ snd $ unzip amazonReviews -- | Building a lexicon (with a threshold) -- ■■■
  putStrLn $ "Number of reviews: " ++ (show $ length amazonReviews)
  putStrLn $ "Number of words: " ++ (show $ length wrds)
  saveDictionary dicFile (DictionaryData wrds) -- | Saving the lexicon
  -- | Reading off one-hot factory from the lexicon
  (DictionaryData dict) <- loadDictionary dicFile -- | Loading the lexicon
  let (oneHot, dicDim) = oneHotFactory dict -- | Building a one-hot vector factory from the lexicon
  putStrLn $ "Dimension of dictionary :" ++ (show $ dicDim)
  -- | training setting
  let iter = 50::Int -- | ■■■
      hypParams = HypParams device dicDim 128 128 [(64,Elu),(24,Elu),(5,Elu)] -- | ■■■
      trainingData = amazonReviews
      ratings = asTensor'' device $ map (\r -> r-1) $ fst $ unzip trainingData
  -- | training loop
  initModel <- sample hypParams
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let dists = stack (Dim 0) $ (flip map) trainingData $ \(_,comment) ->
                  mlpLayer (mlp model) $ last $ lstmLayers False (lstm model) $ map (linearLayer (w_emb model) . asTensor'' device . oneHot) $ comment
        dist2 = softmax (Dim 1) dists
        loss = nllLoss' ratings (logSoftmax (Dim 1) dists)
        --l2loss = sumTensors $ map (\v -> mseLoss v v) $ map toDependent $ flattenParameters model
        --loss' = add loss l2loss
        lossValue = (asValue loss)::Float
    print dist2
    showLoss 1 epoc lossValue
    u <- update model opt loss 5e-1
    return (u, lossValue)
  drawLearningCurve "graph-embedding.png" "Learning Curve" [("",reverse losses)]
  saveParams trainedModel "amazon_embedding.model"
  putStrLn "Training finished."

type AmazonReview = (Float, [T.Text])
data DictionaryData = DictionaryData [T.Text] deriving (Show, G.Generic)

instance A.FromJSON DictionaryData
instance A.ToJSON DictionaryData

saveDictionary :: FilePath -> DictionaryData -> IO()
saveDictionary = Y.encodeFile 

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

data HypParams = HypParams {
  dev :: Device,
  dictDim :: Int,
  embDim :: Int,
  stateDim :: Int,
  mlp_layers :: [(Int,ActName)]
  } deriving (Eq, Show)

data Params = Params {
  w_emb :: LinearParams, 
  lstm :: LstmParams,
  mlp :: MLPParams
  } deriving (G.Generic)

instance Parameterized Params

instance Randomizable HypParams Params where
  sample HypParams{..} = 
    Params
    <$> sample (LinearHypParams dev dictDim embDim)
    <*> sample (LstmHypParams dev stateDim 1)
    <*> sample (MLPHypParams dev stateDim mlp_layers)

{-
parseJsonStream :: FilePath -> IO [[AmazonReview]]
parseJsonStream filePath = do
  jsonStream <- BL.readFile filePath
  return $ J.parseLazyByteString (J.arrayOf J.value) jsonStream 

parseWlpFile :: [
parseWlpFile = 
  allfiles <- listDirectory wipdataDir
  textfiles <- filterM doesFileExist $ map (wipdataDir </>) $ filter ("text_" `L.isPrefixOf`) allfiles -- [FilePath]
  texts <- mapM T.readFile textfiles -- [T.Text]
  let wrds = sortWords 40 $ concat $ map T.words $ concat $ map T.lines texts -- [T.Text]
  print $ length wrds
  saveDictionary dicFile (DictionaryData wrds)
  amazonReviews <- parseJsonStream amazonreviewFile

type BaseForm = T.Text
type POS = T.Text
type Info = (BaseForm,POS)

wlpFiles2info :: [FilePath] -> IO([Info])
wlpFiles2info filepaths = concat <$> mapM wlpFile2Info filepaths

wlpFile2Info :: FilePath -> IO([Info])
wlpFile2Info filepath = do
  wlps <- map (T.splitOn "\t") <$> filter (/= T.empty) <$> tail <$> T.lines <$> T.readFile filepath
  return $ map (\w -> (w!!2,w!!3)) wlps 
-}
