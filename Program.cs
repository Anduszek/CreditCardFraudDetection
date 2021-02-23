using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace CreditCardFraudDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = LoadData(context);

            var pipeline = BuildPipeline(context);

            var model = TrainModel(context, data, pipeline);

            Console.WriteLine("Press enter to finish");
            Console.Read();
        }

        private static IDataView LoadData(
            MLContext context)
        {
            Console.Write("Loading data...");

            var dataPath = Path.Combine(Environment.CurrentDirectory, "creditcard.csv");
            var data = context.Data.LoadFromTextFile<FraudData>(dataPath, hasHeader:true, separatorChar:',');

            Console.WriteLine("done!");

            return data;
        }

        private static IEstimator<ITransformer> BuildPipeline(
            MLContext context)
        {
            Console.Write("Building pipeline...");

            var pipeline = context.Transforms
                .CustomMapping<FraudData, ToLabel>(
                    (input, output) =>
                    {
                        output.Label = input.Class == @"""1""";
                    },
                    "LabelMapping"
                )
                .Append(context.Transforms.Concatenate(
                    "Features",
                    "V1",
                    "V2",
                    "V3",
                    "V4",
                    "V5",
                    "V6",
                    "V7",
                    "V8",
                    "V9",
                    "V10",
                    "V11",
                    "V12",
                    "V13",
                    "V14",
                    "V15",
                    "V16",
                    "V17",
                    "V18",
                    "V19",
                    "V20",
                    "V21",
                    "V22",
                    "V23",
                    "V24",
                    "V25",
                    "V26",
                    "V27",
                    "V28"//,
                    //"Amount"
                ))
                //.AppendCacheCheckpoint(context)
                .Append(context.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "Label",
                    featureColumnName: "Features"));
                

            Console.WriteLine("done!");

            return pipeline;
        }
        
        private static ITransformer TrainModel(
            MLContext context,
            IDataView data,
            IEstimator<ITransformer> pipeline)
        {
            Console.Write("Training the model...");
            var partitions = context.Data.TrainTestSplit(data, testFraction: 0.2);
            var model = pipeline.Fit(partitions.TrainSet);
            var predictions = model.Transform(partitions.TestSet);

            var metrics = context.BinaryClassification.Evaluate(predictions, "Label", "Score");

            Console.WriteLine("done!");

            Console.WriteLine();
            Console.WriteLine($"Model metrics:");
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy}");
            Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve}");
            Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall}");
            Console.WriteLine();

            return model;
        }
    }

    public class FraudData
    {
        [LoadColumn(0)]
        public float Time { get; set; }
        [LoadColumn(1)]
        public float V1 { get; set; }
        [LoadColumn(2)]
        public float V2 { get; set; }
        [LoadColumn(3)]
        public float V3 { get; set; }
        [LoadColumn(4)]
        public float V4 { get; set; }
        [LoadColumn(5)]
        public float V5 { get; set; }
        [LoadColumn(6)]
        public float V6 { get; set; }
        [LoadColumn(7)]
        public float V7 { get; set; }
        [LoadColumn(8)]
        public float V8 { get; set; }
        [LoadColumn(9)]
        public float V9 { get; set; }
        [LoadColumn(10)]
        public float V10 { get; set; }
        [LoadColumn(11)]
        public float V11 { get; set; }
        [LoadColumn(12)]
        public float V12 { get; set; }
        [LoadColumn(13)]
        public float V13 { get; set; }
        [LoadColumn(14)]
        public float V14 { get; set; }
        [LoadColumn(15)]
        public float V15 { get; set; }
        [LoadColumn(16)]
        public float V16 { get; set; }
        [LoadColumn(17)]
        public float V17 { get; set; }
        [LoadColumn(18)]
        public float V18 { get; set; }
        [LoadColumn(19)]
        public float V19 { get; set; }
        [LoadColumn(20)]
        public float V20 { get; set; }
        [LoadColumn(21)]
        public float V21 { get; set; }
        [LoadColumn(22)]
        public float V22 { get; set; }
        [LoadColumn(23)]
        public float V23 { get; set; }
        [LoadColumn(24)]
        public float V24 { get; set; }
        [LoadColumn(25)]
        public float V25 { get; set; }
        [LoadColumn(26)]
        public float V26 { get; set; }
        [LoadColumn(27)]
        public float V27 { get; set; }
        [LoadColumn(28)]
        public float V28 { get; set; }
        [LoadColumn(29)]
        public float Amount { get; set; }
        [LoadColumn(30)]
        public string Class { get; set; }
    }

    public class FromLabel
    {
        public string Class { get; set; }
    }

    public class ToLabel
    {
        public bool Label { get; set; }
    }
}
