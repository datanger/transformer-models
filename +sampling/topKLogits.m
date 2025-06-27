function logits = topKLogits(logits, topK, varargin)
% topKLogits   Return the top K logits
%
%   logits = topKLogits(logits, k) will return a vector of logits where any
%   classes that are not in the top K largest values will be supressed.
%   Values are supressed by setting them to large negative values.
%
%   logits = topKLogits(logits, k, 'Temperature', t) applies temperature
%   scaling before selecting top-k values.
%
%   Inputs:
%       logits  - A numClasses-by-1 vector of logits.
%       k       - The number of values to 'keep'. Everything outside of the
%                 top K values will be supressed. Note for many typical use
%                 cases this parameter can have a big effect.
%       t       - Temperature parameter for scaling logits (default: 1.0)
%
%   Outputs:
%       logits  - A numClasses-by-1 vector of logits.

arguments
    logits
    topK (1,1) {mustBePositive,mustBeInteger}
    varargin.Temperature (1,1) {mustBePositive} = 1.0
end

if isa(logits, 'dlarray')
    extractedLogits = extractdata(logits);
else
    extractedLogits = logits;
end

% Apply temperature scaling
if varargin.Temperature ~= 1.0
    extractedLogits = extractedLogits / varargin.Temperature;
end

% Ensure numerical stability
extractedLogits = extractedLogits - max(extractedLogits);

[~,classRanks] = sort(extractedLogits, 1, 'descend');

notTopKIndices = ( (topK+1):size(classRanks,1) )';
notTopKRows = classRanks(notTopKIndices,:);
notTopKColumns = repmat(1:size(extractedLogits,2),size(notTopKIndices,1),1);

notTopKIndices = sub2ind(size(extractedLogits), notTopKRows, notTopKColumns);

% We want to make sure that when softmax is applied to the logits, the
% probability for the classes that are not in the top K are zero. We do
% this by setting entries that are not in the top K to large negative
% values.
logits( notTopKIndices ) = -1e10;

end