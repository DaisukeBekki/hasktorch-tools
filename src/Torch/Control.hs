module Torch.Control (
  foldLoop,
  foldLoopM,
  mapAccumM,
  trainLoop,
  makeBatch
  ) where

import Control.Monad (foldM)           --base
import Data.List     (foldl')          --base

-- | syntactic sugar for looping with foldl'
foldLoop :: (Foldable t) => t a -> b -> (a -> b -> b) -> b
foldLoop xs zero f = foldl' (flip f) zero xs

-- | syntactic sugar for looping with foldM
foldLoopM :: (Foldable t, Monad m) => t a -> b -> (a -> b -> m b) -> m b
foldLoopM xs zero f = foldM (flip f) zero xs

mapAccumM :: (Monad m, Foldable t) => t a -> b -> (a -> b -> m (b,c)) -> m (b, [c])
mapAccumM xs zero f = do
  foldM (\(prev,lst) x -> do
                          bc <- f x prev
                          return (fst bc, (snd bc):lst)
                          ) (zero,[]) xs

-- | Variant of mapAccumM, specific to deep learning with batched-data
trainLoop :: (Monad m) => [i] -> [d] -> (mod,opt) -> (i -> d -> (mod,opt) -> m ((mod,opt),loss)) -> m ((mod,opt), [loss])
trainLoop epocs dats zero f = do
  foldM (\(prev,lst) (epoc,dat) -> do
                                   bc <- f epoc dat prev
                                   return (fst bc, (snd bc):lst)
                                   ) (zero,[]) $ zip epocs dats

makeBatch :: Int -> [a] -> [[a]]
makeBatch _ [] = []
makeBatch n lst = case splitAt n lst of
  (h,t) -> h:(makeBatch n t)
  

{-
scanLoop :: [a] -> b -> (b -> a -> b) -> [b]
scanLoop xs zero f = scanl f zero xs

scanLoopM :: (Monad m) => [a] -> b -> (a -> b -> m b) -> m [b]
scanLoopM xs zero f = scanlM (flip f) zero xs

scanlM :: (Monad m) => (b -> a -> m b) -> b -> [a] -> m [b]
scanlM f z0 xs = sequenceA $ scanl f' (return z0) xs
  where f' acc x = acc >>= flip f x -- :: m b -> a -> m b
-}
  

