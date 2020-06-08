module Torch.Util.Dict (
  bit,
  bits,
  oneHot,
  Hist,
  pushWords,
  toList,
  Torch.Util.Dict.lookup,
  fetchIndex,
  filterHistByValue,
  labelTensor,
  labels2vec,
  labels2num,
  multilabels2vec,
  ) where

import qualified Data.Char as C         --base
import qualified System.IO    as IO        --base
--import qualified System.FilePath.Posix as IO --filepath
import qualified Data.List    as L       --base
import qualified Data.List.Split as L    --base
import qualified Data.Text    as T    --text
import qualified Data.Text.IO as T    --text
import qualified Data.Map.Strict as M --container
import qualified Data.Maybe as Maybe  --

-- | d次元の{0,1}ベクトルで、n番目のみ1であるものをStringで返す
bit :: Int -> Int -> [Char]
bit dim nth = if nth < 0
                then "1" ++ replicate (dim-1) '0'
                else replicate nth '0' ++ "1" ++ replicate (dim-nth-1) '0'

bits :: Int -> [Int] -> [Char]
bits dim nths = 
  let b = L.foldl' (\acc i -> if i >= dim
                                 then acc
                                 else acc ++ replicate (i - (length acc)) 0 ++ [1]) [] $ L.sort nths in
  map C.intToDigit $ b ++ replicate (dim - length b) 0

-- | d次元の{0,1}（ただしFloat）ベクトルでn番目のみ1.0である[Float]を返す。
oneHot :: Int -> Int -> [Float]
oneHot dim nth = replicate nth (0::Float) ++ ((1::Float):(replicate (dim-nth-1) (0::Float)))

-- Histgram
  
type Hist a = M.Map a Int

pushWords :: (Ord a) => [a] -> Hist a
pushWords = L.foldl' f M.empty
  where f hist key = M.insertWith (+) key (1::Int) hist

filterHistByValue :: (Int -> Bool) -> Hist a -> Hist a
filterHistByValue = M.filter

toList :: (Ord a) => Hist a -> [(a,Int)]
toList = M.toList

lookup :: (Ord a) => a -> Hist a -> Maybe Int
lookup = M.lookup

-- | textがhistのi番目の要素で値（頻度）がjなら[(i,j)]を、histに含まれないなら[]を返す。
fetchIndex :: [(T.Text,Int)] -> T.Text -> [(Int,Int)]
fetchIndex hist text = 
  case do
       i <- L.lookup text hist
       j <- L.elemIndex (text,i) hist
       return (j,i) 
       of Just (j,i) -> [(j,i)]
          Nothing    -> []
    
-- | label set to tensors, returns a hyperparameter (=the number of unique labels)
labelTensor :: Bool -> FilePath -> String -> [T.Text] -> IO(Int)
labelTensor verbose datadir filename labels = do
  let labels_sorted = reverse $ L.sortOn snd $ M.toList $ pushWords labels
      num_of_labels = length labels_sorted
  labels_in_num <- mapM (\label -> do case L.elemIndex label $ fst $ unzip labels_sorted of
                                        Just labelIndex -> return $ L.intersperse ',' $ bit num_of_labels labelIndex
                                        Nothing -> ioError (userError $ T.unpack label ++ " not found.")
                        ) labels
  if verbose 
     then IO.hPutStrLn IO.stderr $ "Number of unique " ++ filename ++ ": " ++ (show num_of_labels)
     else return () 
  --
  --IO.withFile (datadir ++ "/" ++ filename ++ "_sorted.txt") IO.WriteMode (\h -> mapM_ (printPair h) labels_sorted)
  IO.withFile (datadir ++ "/" ++ filename ++ ".txt") IO.WriteMode (\h -> mapM_ (IO.hPutStrLn h) labels_in_num)
  --
  return num_of_labels

-- | 
-- Usage: f anns [0,4,5,6,7] "hoge.txt"はanns中0,4,5,6,7番目のラベルについて、one-hot vectorに
-- 変換したものをつなげてファイル"hoge.txt"に書き出す。引数はベクトルの次元数。
labels2vec :: [[T.Text]] -> [Int] -> FilePath -> IO(Int)
labels2vec anns indices filepath = do
  -- それぞれのindexについてラベル数を数える。
  let vec = map concat $ L.transpose $ map (\i -> let labels = map (!!i) anns
                                                      nubbed = L.nub labels
                                                      len = length nubbed in
                                                  map (bit len) $ Maybe.catMaybes $ map (\l -> L.elemIndex l nubbed) labels
                                           ) indices
  IO.withFile filepath IO.WriteMode (\h -> mapM_ (IO.hPutStrLn h . L.intersperse ',') vec)
  return $ length $ head vec

labels2num :: [[T.Text]] -> [Int] -> FilePath -> IO(Int,Int)
labels2num anns indices filepath = do
  -- それぞれのindexについてラベル数を数える。
  let ianns = map (\i -> map (!!i) anns) indices
      nubbed = L.nub $ L.concat ianns
      vec = L.transpose $ map (\col -> map show $ Maybe.catMaybes $ map (\l -> L.elemIndex l nubbed) col) ianns
  IO.withFile filepath IO.WriteMode (\h -> mapM_ (IO.hPutStrLn h . L.intercalate ",") vec)
  return (length $ head vec, length nubbed)

multilabels2vec :: [[T.Text]] -> Int -> T.Text -> FilePath -> IO(Int)
multilabels2vec anns index delimiter filepath = do
  let labelss = map (\a -> T.splitOn delimiter $ a!!index) anns
      nubbed = L.nub $ L.concat labelss
      len = length nubbed
  IO.withFile (filepath ++ ".sorted") IO.WriteMode (\h ->
    mapM_ (T.hPutStrLn h) nubbed
    )
  IO.withFile filepath IO.WriteMode (\h ->
    mapM_ (\labels -> do
              let nths = Maybe.catMaybes $ map (\l -> L.elemIndex l nubbed) labels
              IO.hPutStrLn h $ L.intersperse ',' $ bits len nths
          ) labelss
    )
  file <- IO.readFile filepath
  return $ length $ L.splitOn "," $ head $ lines file

