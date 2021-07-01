// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace SemanticSearch.Models.DistilBert
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.OnnxRuntime.Tensors;
    using TorchSharp.Tensor;

    public class DistilBert
    {
        private static readonly string[] OutputColumnNames =
        {
            "output_0"
        };

        private static readonly string[] InputColumnNames =
        {
            "input_ids", "attention_mask"
        };

        private readonly DistilBertConfig config;
        private readonly BertTokenizer tokenizer;

        public DistilBert(DistilBertConfig config)
        {
            this.config = config;
            this.tokenizer = new BertTokenizer(File.ReadAllLines("Resources/Models/vocab.txt").ToList());
        }

        public TorchTensor GenerateVectors(IEnumerable<string> input)
        {
            var modelPath = "Resources/Models/msmarco-distilbert-base-v3.onnx";
            var mlContext = new MLContext();

            var inputTexts = input.ToList();
            var batchSize = inputTexts.Count;

            // Onnx models do not support variable dimension vectors. We're using
            // schema definitions to predict a batch.
            // Input schema dimensions: batchSize x sequence
            var inputSchema = SchemaDefinition.Create(typeof(DistilBertInput));
            inputSchema["input_ids"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Int64,
                    batchSize,
                    this.config.MaxSequenceLength);
            inputSchema["attention_mask"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Int64,
                    batchSize,
                    this.config.MaxSequenceLength);

            // Onnx models may have hardcoded dimensions for inputs. Use a custom
            // schema for variable dimension since the number of text documents
            // are a user input for us (batchSize).
            var inputShape = new Dictionary<string, int[]>
            {
                { "input_ids", new[] { batchSize, this.config.MaxSequenceLength } },
                { "attention_mask", new[] { batchSize, this.config.MaxSequenceLength } }
            };
            var pipeline = mlContext.Transforms
                .ApplyOnnxModel(
                    OutputColumnNames,
                    InputColumnNames,
                    modelPath,
                    inputShape,
                    null,
                    true);

            // Setup the onnx model
            var trainingData = mlContext.Data.LoadFromEnumerable(new List<DistilBertInput>(), inputSchema);
            var model = pipeline.Fit(trainingData);

            // Output schema dimensions: batchSize x sequence x 768
            var outputSchema = SchemaDefinition.Create(typeof(DistilBertOutput));
            outputSchema["output_0"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Single,
                    batchSize,
                    this.config.MaxSequenceLength,
                    768);

            var encodedCorpus = this.PrepareInput(inputTexts);
            var engine = mlContext.Model
                .CreatePredictionEngine<DistilBertInput, DistilBertOutput>(
                    model,
                    inputSchemaDefinition: inputSchema,
                    outputSchemaDefinition: outputSchema);
            var predict = engine.Predict(encodedCorpus);

            return Pooling.MeanPooling(
                predict.Embedding,
                encodedCorpus.AttentionMask,
                batchSize,
                this.config.MaxSequenceLength);
        }

        public DistilBertInput PrepareInput(string text)
        {
            return this.Encode(this.tokenizer.Tokenize(new[] { text }), this.config.MaxSequenceLength);
        }

        public DistilBertInput PrepareInput(IEnumerable<string> texts)
        {
            var inputTexts = texts.ToList();
            var batchSize = inputTexts.Count;

            // Encode the inputs with Bert Tokenizer
            var distilBertInputs = inputTexts.Select(text => this.Encode(
                this.tokenizer.Tokenize(new[] { text }),
                this.config.MaxSequenceLength)).ToList();

            // Convert encoded inputs to tensors
            var inputIdsTensor = new DenseTensor<long>(
                distilBertInputs.SelectMany(b => b.InputIds).ToArray(),
                new[]
                {
                    batchSize, this.config.MaxSequenceLength
                });
            var attentionMaskTensor = new DenseTensor<long>(
            distilBertInputs.SelectMany(b => b.AttentionMask).ToArray(),
            new[]
                {
                    batchSize, this.config.MaxSequenceLength
                });

            return new DistilBertInput
            {
                InputIds = inputIdsTensor.ToArray(),
                AttentionMask = attentionMaskTensor.ToArray()
            };
        }

        private DistilBertInput Encode(
            List<(string Token, int Index)> tokens,
            int maxSequenceLength)
        {
            var padding = Enumerable
                .Repeat(0L, maxSequenceLength - tokens.Count)
                .ToList();

            var tokenIndexes = tokens
                .Select(token => (long)token.Index)
                .Concat(padding)
                .ToArray();

            var segmentIndexes = this.GetSegmentIndexes(tokens)
                .Concat(padding)
                .ToArray();

            var inputMask =
                tokens.Select(o => 1L)
                    .Concat(padding)
                    .ToArray();

            return new DistilBertInput
            {
                InputIds = tokenIndexes,
                AttentionMask = inputMask
            };
        }

        private IEnumerable<long> GetSegmentIndexes(
            List<(string Token, int Index)> tokens)
        {
            var segmentIndex = 0;
            var segmentIndexes = new List<long>();

            foreach (var (token, _) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == BertTokenizer.DefaultTokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }
    }
}