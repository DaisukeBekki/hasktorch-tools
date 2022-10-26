{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (when)              --base
import System.FilePath ((</>))           --filepath
import qualified Data.Text as T          --text
import qualified Data.Text.IO as T       --text
import Torch.Tensor (asValue)                    --hasktorch-tools
import Torch.Device (Device(..),DeviceType(..))  --hasktorch-tools
import Torch.Tensor.TensorFactories (asTensor'') --hasktorch-tools
import ML.Util.Directory (getFileList)        --nlp-tools
import ML.Util.Dict (sortWords, oneHotFactory)--nlp-tools
import Torch.LangModel.Juman.Dict (prepareJumanData) --hasktorch-tools
import Torch.Config.JumanLstmConfig            --hasktorch-tools

-- | Juman言語モデル構築

main :: IO ()
main = do
  let debug = True
  (_,config) <- getConfig  -- | 設定ファイル(config.yaml)読み込み。
  let textdataDir = textdata_dirname config -- | テキストデータのディレクトリ名
      jumanlogDir = jumanlog_dirname config -- | juman解析結果を保存しておくディレクトリ名
      dicFile = dic_filename config
  -- | テキストデータ読み込みと変換
  textFiles <- getFileList "txt" textdataDir -- | textdataDirから.txtファイルのみ取り出す
  --when debug $ mapM_ (putStrLn . show) textFiles
  --when debug $ putStrLn $ (show $ length textFiles) ++ " files found."
  {-
  jumanOutput <- processTextsByJuman jumanlogDir textFiles -- | [[JumanData]] .txtファイルをJuman++で処理
  let (kihons,hinsis) = buildDictionary [] $ concat jumanOutput -- | （登場した語のリスト、登場した形態素のリスト）
  J.saveDictionary dicFile $ DictionaryData (sortWords 3 kihons) (sortWords 0 hinsis) -- | 語のリストから辞書を作成し、dicFileに保存
  (J.DictionaryData cKihons cHinsis) <- J.loadDictionary dicFile 
  let (kihon2OneHot, dimKihon) = oneHotFactory cKihons
      (hinsi2OneHot, dimHinsi) = oneHotFactory cHinsis
  putStrLn $ "Number of words: " ++ (show dimKihon)
  putStrLn $ "Number of POS: " ++ (show dimHinsi)
  --print $ asTensor'' dev $ hinsi2OneHot "接尾辞"
-}
  when debug $ putStrLn "finished."

