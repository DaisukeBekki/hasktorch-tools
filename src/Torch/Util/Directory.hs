module Torch.Util.Directory (
  checkDir,
  checkFile,
  getFileList
  ) where

import Control.Monad (when,filterM)      --base
import Control.Exception (throw)         --base
import System.Directory (doesDirectoryExist, doesFileExist, listDirectory) --directory
import System.FilePath ((</>),isExtensionOf)   --filepath

checkDir :: FilePath -> IO()
checkDir filepath = do
  let whenM s r = s >>= flip when r
  whenM (not <$> doesDirectoryExist filepath) $ throw (userError $ filepath ++ " does not exist.") 

checkFile :: FilePath -> IO()
checkFile filepath = do
  let whenM s r = s >>= flip when r
  whenM (not <$> doesFileExist filepath) $ throw (userError $ filepath ++ " does not exist.") 

-- | filepathはフルパスを仮定
getFileList :: String -> FilePath -> IO([FilePath])
getFileList ext filepath = do 
  checkDir filepath 
  fileDirList <- map (filepath </>) <$> listDirectory filepath -- | ファイル・ディレクトリのリストを得る
  dirs <- filterM doesDirectoryExist fileDirList
  files <- filter (isExtensionOf ext) <$> filterM doesFileExist fileDirList
  files_subdir <- mapM (getFileList ext) dirs
  return $ concat $ files:files_subdir

