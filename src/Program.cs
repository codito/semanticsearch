// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch
{
    using System;
    using System.Linq;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using SemanticSearch.Datasets;
    using SemanticSearch.Models;
    using SemanticSearch.Models.DistilBert;

    internal class Program
    {
        private static void Main()
        {
            var context = new MLContext();
            var dataF = context.Data.LoadFromTextFile<Movie>(
                    "Resources/Datasets/tmdb_5000.csv",
                    '|',
                    true);

            // var data = context.Data.TakeRows(context.Data.ShuffleRows(dataF), 10);
            var data = context.Data.TakeRows(dataF, 100);
            var dataOverview = data
                .GetColumn<string>("Overview")
                .Select((s, i) => (s, i))
                .ToDictionary(kv => kv.i, kv => kv.s);
            var dataTitle = data.GetColumn<string>("Title")
                .Select((s, i) => (s, i))
                .ToDictionary(kv => kv.i, kv => kv.s);

            var bert = new DistilBert(new DistilBertConfig());
            var corpusEmbeddings = bert.GenerateVectors(dataOverview.Values);

            Console.WriteLine("Enter the movie query: ");
            var query = new[] { Console.ReadLine() };
            var queryEmbeddings = bert.GenerateVectors(query);

            var topK = Similarity.TopKByCosineSimilarity(
                corpusEmbeddings,
                queryEmbeddings,
                5);

            var scores = topK.Values.Data<float>().GetEnumerator();
            foreach (var index in topK.Indexes.Data<long>().ToArray())
            {
                scores.MoveNext();
                Console.WriteLine($"Movie: {dataTitle[(int)index]}");
                Console.WriteLine($"\tScore: {scores.Current * 100}");
                Console.WriteLine($"\tOverview: {dataOverview[(int)index]}");
                Console.WriteLine();
            }
        }
    }
}
