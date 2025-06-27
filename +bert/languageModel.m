function z = languageModel(x,p,varargin)
% languageModel   The BERT language model.
%
%   Z = bert.languageModel(X,parameters) performs inference with a BERT model
%   on the input X, and applies the output layer projection onto the
%   associated vocabulary. The input X is a 1-by-numInputTokens-by-numObs
%   array of encoded tokens. The return is an array Z of size
%   vocabularySize-by-numInputTokens-by-numObs. In particular the language model is
%   trained to predict a reasonable word for each masked input token.
%
%   Z = bert.languageModel(X,parameters,'UseCache',true) enables caching
%   for improved performance on repeated inputs.

% Copyright 2021 The MathWorks, Inc.
arguments
    x
    p
    varargin.UseCache (1,1) logical = false
end

if ~isfield(p.Weights,'masked_LM')
    error("bert:languageModel:MissingLMWeights","Parameters do not include masked_LM weights");
end

% Check cache for repeated inputs
persistent cache
if varargin.UseCache && ~isempty(cache)
    inputHash = hashInput(x);
    if isfield(cache, inputHash)
        z = cache.(inputHash);
        return;
    end
end

z = bert.model(x,p);
z = bert.layer.languageModelHead(z,p.Weights.masked_LM,p.Weights.embeddings.word_embeddings);

% Store in cache
if varargin.UseCache
    if isempty(cache)
        cache = struct();
    end
    inputHash = hashInput(x);
    cache.(inputHash) = z;
end
end

function h = hashInput(x)
% Simple hash function for input caching
if isa(x, 'dlarray')
    x = extractdata(x);
end
h = sprintf('input_%d_%d_%d', size(x, 1), size(x, 2), size(x, 3));
end