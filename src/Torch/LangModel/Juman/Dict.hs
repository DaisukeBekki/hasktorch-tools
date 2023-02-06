{-# LANGUAGE ExtendedDefaultRules, DeriveGeneric #-}

-- Daisuke Bekki

-- | forked from dynet-tools/Tools/Juman.hs
module Torch.LangModel.Juman.Dict (
  prepareJumanData,
  dictionary2oneHot,
  WordInfo,
  jumanData2Tuples
  ) where

import Control.Monad (when)        --base
import System.FilePath ((</>),(<.>),takeBaseName) --filepath
import Control.Exception (throw)        --base
import Control.Monad (when,forM,guard)  --base
import System.Directory (doesFileExist,doesDirectoryExist) --base
import qualified GHC.Generics as G --base
import qualified Data.List as L    --base
import qualified GHC.Generics as G --base
import qualified Data.Text as T    --text
import qualified Data.Text.IO as T --text
import qualified Shelly as S       --shelly
import qualified Data.Aeson as A   --aeson
import qualified Data.ByteString.Char8 as B --bytestring 
import qualified Data.Yaml             as Y --yaml
import Text.Juman (JumanData(..),file2jumanLine,jumanParser) --nlp-tools
import ML.Util.Directory (getFileList,checkFile)             --nlp-tools
import qualified ML.Util.Dict as D                           --nlp-tools

prepareJumanData :: Bool -> [POS] -> Int -> Int -> FilePath -> FilePath -> FilePath -> IO()
prepareJumanData showStat posses baseFormThreshold posThreshold textdataDir jumanlogDir dicFile = do
  txtList <- getFileList "txt" textdataDir -- | textdataDirから.txtファイルのみ取り出す
  when showStat $ putStrLn $ "Number of Text files: " ++ (show $ length txtList)
  jumanOutput <- concat <$> processTextsByJuman jumanlogDir txtList -- | [[JumanData]] .txtファイルをJuman++で処理
  saveDictionary dicFile $ buildDictionary posses baseFormThreshold posThreshold jumanOutput -- | （登場した語のリスト、登場した形態素のリスト）語のリストから辞書を作り保存
  putStrLn $ "Juman data saved in: " ++ (show dicFile)

dictionary2oneHot :: FilePath -> IO (WordInfo -> [Float],Int)
dictionary2oneHot dicFile = oneHotFactory <$> loadDictionary dicFile

type SurfaceForm = T.Text
type BaseForm = T.Text
type POS = T.Text
type WordInfo = (SurfaceForm,BaseForm,POS)

printWordInfo :: WordInfo -> IO()
printWordInfo (surfaceForm,baseForm,pos) = do
  putStr "("
  T.putStr surfaceForm
  putStr ","
  T.putStr baseForm
  putStr ","
  T.putStr pos
  putStrLn ")"

-- | takes an output filepath and a list of texts,
--   process each text with `callJuman',
--   and write the result in a specified fiie (unless it already exists)
processTextsByJuman :: FilePath -> [FilePath] -> IO [[JumanData]]
processTextsByJuman outputDirPath inputFilePaths = S.shelly $ do
  dir_exist <- S.test_d outputDirPath
  when (not dir_exist) $ S.echo_n_err $ T.pack $ outputDirPath ++ " is not a directory." 
  let outputDir = S.fromText $ T.pack outputDirPath
  forM inputFilePaths $ \inputFilePath -> do
    let outputFile = outputDir </> (takeBaseName inputFilePath) <.> "juman"
    file_exist <- S.test_f outputFile
    jumanLines <- if file_exist -- if the output file already exists 
      then do
        --S.echo_n_err $ T.pack $ outputFile
        --                        ++ " exits. To re-run juman over the text files, delete "
        --                        ++ outputFile
        --                        ++ " and execute this program again.\n"
        S.echo_n_err "x"
        S.readfile outputFile
      else do
        jumanLines <- file2jumanLine inputFilePath
        S.writefile outputFile jumanLines 
        S.echo_n_err "o" 
        return jumanLines
    return $ map jumanParser $ filter (/= T.empty) $ T.lines jumanLines

data DictionaryData = DictionaryData {
  baseForms :: [BaseForm],
  poss :: [POS]
  } deriving (Show, G.Generic)

instance A.FromJSON DictionaryData
instance A.ToJSON DictionaryData

-- | Create a pair (or a tuple) of sorted labels from juman-processed texts
--   ([k1,...,kn],[p1,...,pn])
buildDictionary :: [POS]   -- ^ poss to use.  when posfilter==[], all poss are used.
                   -> Int  -- ^ counts only baseForms with more than this frequency
                   -> Int  -- ^ counts only posses with more than this frequency
                   -> [JumanData]    -- ^ All the juman output data
                   -> DictionaryData
buildDictionary posfilter baseFormThreshold posThreshold jumanData =
  let (ks,ps) = unzip $ do -- list monad
                        (_,k,p) <- jumanData2Tuples jumanData
                        -- Only considers elements specified in posfilter
                        S.when (posfilter /= []) $ guard $ L.elem p posfilter
                        return (k,p)        -- [(k1,p1),...,(kn,pn)]
  in DictionaryData (D.sortWords baseFormThreshold ks) (D.sortWords posThreshold ps)

jumanData2Tuples :: [JumanData] -> [WordInfo]
jumanData2Tuples jumanData = map jumanData2Tuple $ filter isContent jumanData 

jumanData2Tuple :: JumanData -> WordInfo
jumanData2Tuple jumanData = case jumanData of
  (JumanWord hyoso _ kihon hinsi _ _ _ _ _ _ _ _) -> (hyoso,kihon,hinsi)
  (AltWord _ _ _ _ _ _ _ _ _ _ _ _) -> ("ALT","ALT","ALT")
  EOS     -> ("EOS","EOS","EOS")
  Err _ _ -> ("ERR","ERR","ERR")

isContent :: JumanData -> Bool
isContent jumanData = case jumanData of
  (JumanWord _ _ _ _ _ _ _ _ _ _ _ _) -> True
  (AltWord _ _ _ _ _ _ _ _ _ _ _ _) -> False
  EOS     -> True
  Err _ _ -> False

saveDictionary :: FilePath -> DictionaryData -> IO()
saveDictionary = Y.encodeFile 

loadDictionary :: FilePath -> IO(DictionaryData)
loadDictionary filepath = do
  checkFile filepath
  content <- B.readFile filepath
  let parsedDic = Y.decodeEither' content :: Either Y.ParseException DictionaryData
  case parsedDic of
    Left parse_exception -> error $ "Could not parse dic file " ++ filepath ++ ": " ++ (show parse_exception)
    Right dic -> return dic

oneHotFactory :: DictionaryData -> (WordInfo -> [Float],Int)
oneHotFactory DictionaryData{..} = 
  let (oneHotB,dimB) = D.oneHotFactory baseForms
      (oneHotP,dimP) = D.oneHotFactory poss
  in (\(_,baseForm,pos) -> (oneHotB baseForm) ++ (oneHotP pos), dimB + dimP)


{-
-- | Remove (from a given list) elements whose occurrence is equal or less than the threshold
cutLessFreqWords :: Int      -- ^ The minimum number of occurences of a word (otherwise the word is ignored)
                 -> [T.Text] -- ^ a list of words
                 -> [T.Text]
cutLessFreqWords threshold wds = fst $ unzip $ reverse $ L.sortOn snd $ U.toList $ U.filterHistByValue (>= threshold) $ U.pushWords wds
-}

