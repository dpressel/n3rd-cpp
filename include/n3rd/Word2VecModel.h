//
// Created by Daniel on 9/25/2015.
//

#ifndef __N3RD_CPP_WORD2VECMODEL_H__
#define __N3RD_CPP_WORD2VECMODEL_H__

#include <vector>
#include <memory>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <map>
 namespace n3rd
 {
     struct Word2VecModel
     {
         enum
         {
             MAX_W = 50
         };

         Word2VecModel(long long words, long long size) :
                 numWords(words), embedSz(size)
         {

             M.resize((long long) words * (long long) size);

         }

         ~Word2VecModel()
         {
         }

         std::map<std::string, long> vocab;
         std::vector<float> M;
         long long numWords;
         long long embedSz;

         std::vector<float> getVec(std::string word)
         {
             auto it = vocab.find(word);
             if (it == vocab.end())
             {
                 return std::vector<float>();
             }

             float *a = &(M[it->second * embedSz]);

             return std::vector<float>(a, a + (long) embedSz);
         }

         static Word2VecModel* loadWord2VecModel(std::string file);
     };

 }
#endif
