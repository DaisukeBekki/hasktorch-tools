{-# LANGUAGE ExtendedDefaultRules, DeriveGeneric #-}
{-# OPTIONS_GHC -fno-warn-type-defaults #-}

module Torch.Config.JumanLstmConfig (
  Config(..),
  readConfig,
  getConfig
  ) where

import qualified GHC.Generics          as G --base
import qualified Data.Text             as T --text
import qualified Data.Aeson            as A --aeson
import qualified Data.ByteString.Char8 as B --bytestring 
import qualified Data.Yaml             as Y --yaml
import qualified System.Directory      as D --directory
import System.Environment (getArgs)  --base
import Control.Exception (throw)     --base
import Control.Monad (when)          --base
default(T.Text)

-- | プログラムに実行環境を指定するyamlファイルを渡す。
-- | そのyamlファイルの仕様を以下で定義。
data Config = Config 
  { textdata_dirname ::  FilePath -- | コーパス（.txtファイル）を置くディレクトリ
  , jumanlog_dirname :: FilePath  -- | Juman解析後の出力（.jumanファイル）を置くディレクトリ
  , dic_filename :: FilePath      -- | 辞書ファイル（yamlファイル）
  } deriving (Show, G.Generic)

instance A.FromJSON Config

readConfig :: FilePath -> IO(Config)
readConfig filepath = do
  whenM (not <$> D.doesFileExist filepath) $
    throw (userError $ filepath ++ " does not exist.")
  content <- B.readFile filepath
  let parsedConfig = Y.decodeEither' content :: Either Y.ParseException Config
  case parsedConfig of
    Left parse_exception -> error $ "Could not parse config file " ++ filepath ++ ": " ++ (show parse_exception)
    Right config -> return config
  where whenM s r = s >>= flip when r

getConfig :: IO(FilePath,Config)
getConfig = do
  (filepath:_) <- getArgs
  config <- readConfig(filepath)
  return (filepath, config)


