//
// Created by Daniel on 10/1/2015.
//

#include "n3rd/Word2VecModel.h"

using namespace n3rd;

Word2VecModel* Word2VecModel::loadWord2VecModel(std::string file)
{

    FILE* f = fopen(file.c_str(), "rb");
    assert(f != nullptr);
    long long words, size, a, b, c, d, cn, bi[100];

    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);

    Word2VecModel* model = new Word2VecModel(words, size);
    a = 0;
    float len;
    for (b = 0; b < words; b++) {
        a = 0;

        char buf[Word2VecModel::MAX_W] = "";
        while (1) {

            // b is the word offset
            buf[a] = (char)fgetc(f);
            if (feof(f) || (buf[a] == ' ')) break;
            if ((a < Word2VecModel::MAX_W) && (buf[a] != '\n')) a++;
        }
        buf[a] = 0;

        std::string str(buf);
        model->vocab[str] = b;
        for (a = 0; a < size; a++) fread(&(model->M[a + b * size]), sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += model->M[a + b * size] * model->M[a + b * size];
        len = std::sqrt(len);
        for (a = 0; a < size; a++) model->M[a + b * size] /= len;
    }
    fclose(f);
    return model;
}