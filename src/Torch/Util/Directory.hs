module Torch.Util.Directory (
  checkDir,
  checkFile,
  getFileList
  ) where

import Control.Monad (when,filterM)      --base
import Control.Exception (throw)         --base
import System.Directory (doesDirectoryExist, doesFileExist, listDirectory) --directory
import System.FilePath ((</>),isExtensionOf)   --filepath

-- | filepathが存在するディレクトリであることを確認する。
checkDir :: FilePath -> IO()
checkDir filepath = do
  let whenM s r = s >>= flip when r
  whenM (not <$> doesDirectoryExist filepath) $ throw (userError $ filepath ++ " does not exist.") 

-- | filepathが存在するファイルであることを確認する。
checkFile :: FilePath -> IO()
checkFile filepath = do
  let whenM s r = s >>= flip when r
  whenM (not <$> doesFileExist filepath) $ throw (userError $ filepath ++ " does not exist.") 

-- | filepath以下のディレクトリ（再帰的）にある拡張子extの
-- | ファイルのリスト（フルパス）を得る。filepathはフルパスで指定すること。
getFileList :: String -> FilePath -> IO([FilePath])
getFileList ext filepath = do 
  checkDir filepath 
  fileDirList <- map (filepath </>) <$> listDirectory filepath -- | ファイル・ディレクトリのリストを得る
  dirs <- filterM doesDirectoryExist fileDirList
  files <- filter (isExtensionOf ext) <$> filterM doesFileExist fileDirList
  files_subdir <- mapM (getFileList ext) dirs
  return $ concat $ files:files_subdir

