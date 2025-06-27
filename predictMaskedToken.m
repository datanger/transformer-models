function out = predictMaskedToken(mdl,str,varargin)
% predictMaskedToken   Given a BERT language model, predict the most likely
% tokens for masked tokens.
%
%   out = predictMaskedToken(mdl, text) returns the string out which
%   replaces instances of mdl.Tokenizer.MaskToken in the string text with
%   the most likely token according to the BERT model mdl.
%
%   out = predictMaskedToken(mdl, text, 'TopK', k) applies top-k sampling
%   to improve prediction diversity.
%
%   out = predictMaskedToken(mdl, text, 'Temperature', t) applies temperature
%   scaling to the logits before sampling.

% Copyright 2021-2023 The MathWorks, Inc.
arguments
    mdl {mustBeA(mdl,'struct')}
    str {mustBeText}
    varargin.TopK (1,1) {mustBePositive,mustBeInteger} = 1
    varargin.Temperature (1,1) {mustBePositive} = 1.0
end
str = string(str);
inSize = size(str);
str = str(1:end);
[seqs,pieces] = arrayfun(@(s)encodeScalarString(mdl.Tokenizer,s),str,'UniformOutput',false);
x = padsequences(seqs,2,'PaddingValue',mdl.Tokenizer.FullTokenizer.encode(mdl.Tokenizer.PaddingToken));
maskCode = mdl.Tokenizer.FullTokenizer.encode(mdl.Tokenizer.MaskToken);
ismask = x==maskCode;
x = dlarray(x);
probs = bert.languageModel(x,mdl.Parameters);

% Apply temperature scaling
if varargin.Temperature ~= 1.0
    logits = log(probs + eps);
    scaledLogits = logits / varargin.Temperature;
    probs = softmax(scaledLogits, 'DataFormat', 'CTB');
end

% Apply top-k filtering
if varargin.TopK > 1
    for i = 1:size(probs, 2)
        for j = 1:size(probs, 3)
            if any(ismask(:, i, j))
                maskedProbs = probs(:, i, j);
                filteredProbs = sampling.topKLogits(maskedProbs, varargin.TopK);
                probs(:, i, j) = filteredProbs;
            end
        end
    end
end

maskedProbs = extractdata(probs(:,ismask));
[~,sampleIdx] = max(maskedProbs,[],1);
predictedTokens = mdl.Tokenizer.FullTokenizer.decode(sampleIdx);
out = strings(numel(seqs),1);
numMaskPerSeq = sum(ismask,2);
maskStartIdx = 1;
for i = 1:numel(seqs)
    startIdx = maskStartIdx;
    maskStartIdx = maskStartIdx+numMaskPerSeq(i);
    out(i) = rebuildScalarString(pieces{i},predictedTokens(startIdx:(startIdx+numMaskPerSeq(i)-1)));
end
out = reshape(out,inSize);
end

function [x,pieces] = encodeScalarString(tok,str)
pieces = split(str,tok.MaskToken);
fulltok = tok.FullTokenizer;
maskCode = fulltok.encode(tok.MaskToken);
x = [];

for i = 1:numel(pieces)
    tokens = fulltok.tokenize(pieces(i));
    if ~isempty(tokens{1})
        % "" tokenizes to empty - awkward
        x = cat(2,x,fulltok.encode(tokens{1}));
    end
    if i<numel(pieces)
        x = cat(2,x,maskCode);
    end
end
x = [fulltok.encode(tok.StartToken),x,fulltok.encode(tok.SeparatorToken)];
end

function out = rebuildScalarString(pieces,predictedTokens)
out = "";
for i = 1:(numel(pieces)-1)
    out = strcat(out,pieces(i),predictedTokens(i));
end
out = strcat(out,pieces(end));
end