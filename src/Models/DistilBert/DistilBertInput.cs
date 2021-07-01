// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch.Models.DistilBert
{
    using Microsoft.ML.Data;

    public class DistilBertInput
    {
        // See https://huggingface.co/transformers/model_doc/distilbert.html
        // Dimensions: batch, sequence
        [VectorType(1, 256)]
        [ColumnName("input_ids")]
        public long[] InputIds { get; set; }

        // Dimensions: batch, sequence
        [VectorType(1, 256)]
        [ColumnName("attention_mask")]
        public long[] AttentionMask { get; set; }
    }
}