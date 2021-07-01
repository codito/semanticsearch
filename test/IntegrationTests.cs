// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch.Tests
{
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using SemanticSearch.Models;
    using SemanticSearch.Models.DistilBert;

    [TestClass]
    public class IntegrationTests
    {
        [TestMethod]
        public void ValidateSimilarity()
        {
            var corpus = new[]
            {
                "What is the capital of France?",
                "How many people live in New York City?"
            };
            var query = new[] { "Paris is the capital city of France. " };
            var bert = new DistilBert(new DistilBertConfig());

            var corpusEmbeddings = bert.GenerateVectors(corpus);
            var queryEmbeddings = bert.GenerateVectors(query);
            var topK = Similarity.TopKByCosineSimilarity(
                corpusEmbeddings,
                queryEmbeddings,
                1);

            Assert.IsTrue(topK.Values.Data<float>()[0] > 0.80);
            Assert.AreEqual(0, topK.Indexes.Data<long>()[0]);
        }
    }
}