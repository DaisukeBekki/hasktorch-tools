module Torch.Util.TSV (
  openTSV,
  kFold,
  ) where

import qualified System.IO    as IO        --base
import qualified System.FilePath.Posix as IO --filepath
import qualified Data.List    as L       --base
import qualified Data.Text    as T    --text
import qualified Data.Text.IO as T    --text

-- | デリミタとCSVファイル（へのパス）を与えると、読み込んで行分割とsplitをしてくれる。
-- また、データ欠損がないかチェック。
openTSV :: T.Text -> FilePath -> IO ([[T.Text]])
openTSV delim filepath = do
  tsv <- T.readFile filepath
  let lns = T.lines tsv
  -- | Data check 1: ファイルが0行なら標準出力にエラーメッセージを返す。
  if null lns
     then ioError (userError $ IO.takeFileName filepath ++ " empty.")
     else return ()
  -- | Split: 1行目は項目の並びとみなす。
  let allcontents@(header:contents) = map (T.splitOn delim) lns
      headerSize = length header
  -- | Data check 2: 各行の要素数が一行目と同じかチェック。
  if any (==headerSize) $ map length contents
     then IO.hPutStrLn IO.stderr $ IO.takeFileName filepath ++ " well-formed: " ++ show headerSize ++ " columns"
     else ioError (userError $ IO.takeFileName filepath ++ " not well-formed.")
  return allcontents

-- | 与えられたfileをK分割する関数
kFold :: Int -> FilePath -> IO([T.Text],[T.Text])
kFold k filepath = do
  content <- T.readFile filepath 
  let (_,train,test) = L.foldl' (\(i,train',test') l -> if mod i k == 0
                                                            then (i+1, train', l:test')
                                                            else (i+1, l:train', test')     
                                ) (0,[],[]) $ T.lines content
  return (train,test)


