// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch.Models.DistilBert
{
    using Microsoft.ML.Data;

    public class DistilBertOutput
    {
        // See https://huggingface.co/transformers/model_doc/distilbert.html
        // Dimensions: batch, sequence, hidden_size
        [VectorType(1, 256, 768)]
        [ColumnName("output_0")]
        public float[] Embedding { get; set; }
    }
}