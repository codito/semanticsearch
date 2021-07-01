// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.
//
// Original source code from
// https://github.com/GerjanVlot/BERT-ML.NET/blob/master/Microsoft.ML.Models.BERT/Extensions/StringExtensions.cs
// Copyright (c) BERT-ML.NET, MIT License

namespace SemanticSearch.Models
{
    using System.Collections.Generic;

    internal static class StringExtensions
    {
        public static IEnumerable<string> SplitAndKeep(
            this string s,
            params char[] delimiters)
        {
            int start = 0, index;

            while ((index = s.IndexOfAny(delimiters, start)) != -1)
            {
                if (index - start > 0)
                {
                    yield return s.Substring(start, index - start);
                }

                yield return s.Substring(index, 1);

                start = index + 1;
            }

            if (start < s.Length)
            {
                yield return s.Substring(start);
            }
        }
    }
}