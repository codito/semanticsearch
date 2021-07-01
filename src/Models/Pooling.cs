// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch.Models
{
    using TorchSharp.Tensor;

    public static class Pooling
    {
        public static TorchTensor MeanPooling(float[] embeddings, long[] attentionMask, long batchSize, long sequence)
        {
            var hiddenSize = 768L;

            // See https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v3#usage-huggingface-transformers
            // Note how the python code below translates to dotnet, thanks to the
            // awesome TorchSharp library.
            //
            // def mean_pooling(model_output, attention_mask):
            //  # First element of model_output contains all token embeddings
            //  token_embeddings = model_output[0]
            //  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            //  sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            //  sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            //  return sum_embeddings / sum_mask
            var tokenEmbeddings = Float32Tensor.from(
                embeddings,
                new[] { batchSize, sequence, hiddenSize });
            var attentionMaskExpanded = Int64Tensor.from(
                    attentionMask,
                    new[] { batchSize, sequence })
                .unsqueeze(-1).expand(tokenEmbeddings.shape).@float();

            var sumEmbeddings = (tokenEmbeddings * attentionMaskExpanded).sum(new[] { 1L });
            var sumMask = attentionMaskExpanded.sum(new[] { 1L }).clamp(1e-9, float.MaxValue);

            return sumEmbeddings / sumMask;
        }

#if NONE
        // Attempt to use NumSharp for mean pooling with numpy like code
        private static float[] MeanPoolingX(float[] embedding, long[] attentionMask)
        {
            // Python code:
            // def mean_pooling(model_output, attention_mask):
            //  token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            //  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            //  sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            //  sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            //  return sum_embeddings / sum_mask
            var batchSize = 2;
            var sequence = 11;
            var hiddenSize = 768;
            var mask = np.ndarray(
                new Shape(batchSize, sequence),
                typeof(long),
                attentionMask).astype(typeof(float));
            var tokenEmbeddings =
                np.ndarray(
                    new Shape(batchSize, sequence, hiddenSize),
                    typeof(float),
                    embedding);
            var expandedMask = np.broadcast_to(np.expand_dims(mask, -1), new Shape(batchSize, sequence, hiddenSize));
            var sumEmbeddings = np.sum(tokenEmbeddings * expandedMask, 1);

            // result[i][k] = sum(T[i][j][k] for j in range(T.shape[1])) for all i,k
            var sum = np.ndarray(new Shape(batchSize, hiddenSize));
            for (int i = 0; i < batchSize; i++)
            {
                for (int k = 0; k < hiddenSize; k++)
                {
                    for (int j = 0; j < sequence; j++)
                    {
                        sum[i, k] += expandedMask[i, j, k];
                    }
                }
            }

            var sumMask = np.clip(
                sum,
                a_min: np.ndarray(new Shape(1), buffer: new[] { 1e-9f }),
                a_max: null);

            return (sumEmbeddings / sumMask).ToArray<float>();
         }
#endif
    }
}