// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch.Models.DistilBert
{
    public class DistilBertConfig
    {
        public DistilBertConfig()
        {
            this.MaxSequenceLength = 256;
        }

        public int MaxSequenceLength { get; set; }
    }
}