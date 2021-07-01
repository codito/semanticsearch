// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch.Tests
{
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using SemanticSearch.Models.DistilBert;

    [TestClass]
    public class DistilBertTests
    {
        [TestMethod]
        public void ShouldEncodeVectorsWithPadding()
        {
            var input = new[]
            {
                "This is a short sequence.",
                "This is a rather long sequence. It is at least longer than the sequence A."
            };
            var model = new DistilBert(new DistilBertConfig { MaxSequenceLength = 19 /* max words */ });

            var feature = input.Select(i => model.PrepareInput(i)).ToList();

            // See tools/tokenizer.py for python code to find expected tokens and attentions
            var inputIds = new[]
            {
                new long[]
                {
                    101, 2023, 2003, 1037, 2460, 5537, 1012, 102, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0
                },
                new long[]
                {
                    101, 2023, 2003, 1037, 2738, 2146, 5537, 1012, 2009, 2003,
                    2012, 2560, 2936, 2084, 1996, 5537, 1037, 1012, 102
                }
            };
            var attentionMasks = new[]
            {
                new long[]
                {
                    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                },
                new long[]
                {
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                }
            };

            CollectionAssert.AreEqual(inputIds[0], feature[0].InputIds);
            CollectionAssert.AreEqual(inputIds[1], feature[1].InputIds);
            CollectionAssert.AreEqual(attentionMasks[0], feature[0].AttentionMask);
            CollectionAssert.AreEqual(attentionMasks[1], feature[1].AttentionMask);
        }
    }
}