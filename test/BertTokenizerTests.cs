// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch.Tests
{
    using System.IO;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using SemanticSearch.Models;

    [TestClass]
    public class BertTokenizerTests
    {
        [TestMethod]
        public void ShouldTokenize()
        {
            var vocab = File.ReadAllLines("Resources/Models/vocab.txt").ToList();
            var tokenizer = new BertTokenizer(vocab);

            var x = tokenizer.Tokenize(new[] { "A Titan RTX has 24GB of VRAM" });

            var expected = new[]
            {
                ("[CLS]", 101), ("a", 1037), ("titan", 16537), ("rt", 19387),
                ("##x", 2595), ("has", 2038), ("24", 2484), ("##gb", 18259),
                ("of", 1997), ("vr", 27830), ("##am", 3286), ("[SEP]", 102)
            };
            CollectionAssert.AreEqual(expected, x);
        }
    }
}
