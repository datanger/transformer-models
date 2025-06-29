%% Predict Masked Tokens Using BERT
% This example shows how to predict masked tokens using a pretrained BERT
% model.
%
% BERT models are trained to perform various tasks. One of the tasks is
% known as masked language modeling which is the task of predicting tokens
% in text that have been replaced by a mask value.
%
% This example shows how to predict masked tokens for text data and
% calculate the token probabilities using a pretrained BERT model.

%% Load Pretrained BERT Model
% Load a pretrained BERT model using the |bert| function. The model
% consists of a tokenizer that encodes text as sequences of integers, and a
% structure of parameters.
mdl = bert

%%
% View the BERT model tokenizer. The tokenizer encodes text as sequences of
% integers and holds the details of padding, start, separator and mask
% tokens.
tokenizer = mdl.Tokenizer

%% Predict Masked Token
% Create a string containing a piece of text and replace a single word with
% the tokenizer mask token.
str = "Text Analytics Toolbox includes tools for preprocessing raw text from sources such as equipment logs, news feeds, surveys, operator reports, and social media.";
strMasked = replace(str,"sources",tokenizer.MaskToken)

%%
% Predict the masked token using the |predictMaskedToken| function. The
% function returns the original string with the mask tokens replaced.
sentencePred = predictMaskedToken(mdl,strMasked)

%% Calculate Prediction Scores with Top-K Sampling
% To get the prediction scores for each word in the model vocabulary, you
% can evaluate the language model directly using the |bert.languageModel|
% function.

%%
% First, tokenize the input sentence with the BERT model tokenizer using
% the |tokenize| function. Note that the tokenizer may split single words
% and also prepends a [CLS] token and appends a [SEP] token to the input.
tokens = tokenize(tokenizer,str);
tokens{1}

%%
% Replace one of the tokens with the mask token.
idx = 16;
tokens{1}(idx) = tokenizer.MaskToken

%%
% Encode the tokens using the BERT model tokenizer using the |encodeTokens|
% function.
X = encodeTokens(tokenizer,tokens);
X{1}

%%
% To get the predictions scores from for the encoded tokens, evaluate the
% BERT language model directly using the |bert.languageModel| function. The
% language model output is a VocabularySize-by-SequenceLength array.
scores = bert.languageModel(X{1},mdl.Parameters);

%%
% Apply top-k sampling to improve prediction diversity
k = 5;
maskedScores = scores(:,idx);
filteredScores = sampling.topKLogits(maskedScores, k);

%%
% View the tokens of the BERT model vocabulary corresponding to the top 10
% prediction scores for the mask token.
[~,idxTop] = maxk(extractdata(filteredScores),10);
tbl = table;
tbl.Token = arrayfun(@(x) decode(tokenizer,x), idxTop);
tbl.Score = filteredScores(idxTop,idx)
