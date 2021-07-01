// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch.Datasets
{
    using Microsoft.ML.Data;

    public class Movie
    {
        [LoadColumn(0)]
        public string Title { get; set; }

        [LoadColumn(1)]
        public string Overview { get; set; }

        [LoadColumn(2)]
        public string Genres { get; set; }

        [LoadColumn(3)]
        public string Keywords { get; set; }
    }
}