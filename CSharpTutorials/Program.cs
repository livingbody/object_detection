using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;


namespace Microsoft.ML.OnnxRuntime.FasterRcnnSample
{
    class Program
    {
        public static void Main(string[] args)
        {


            // Read paths
            string modelFilePath = @"C:\Users\livingbody\source\repos\CSharpTutorials\CSharpTutorials\FasterRCNN-12-int8.onnx";
            string imageFilePath = @"C:\Users\livingbody\source\repos\CSharpTutorials\CSharpTutorials\a.jpg";
            string outImageFilePath = "C:\\Users\\livingbody\\source\\repos\\CSharpTutorials\\CSharpTutorials\\outputs.jpg";


            // Read image
            using Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>(imageFilePath);

            // Resize image
            float ratio = 800f / Math.Min(image.Width, image.Height);
            image.Mutate(x => x.Resize((int)(ratio * image.Width), (int)(ratio * image.Height)));

            // Preprocess image
            var paddedHeight = (int)(Math.Ceiling(image.Height / 32f) * 32f);
            var paddedWidth = (int)(Math.Ceiling(image.Width / 32f) * 32f);
            Tensor<float> input = new DenseTensor<float>(new[] { 3, paddedHeight, paddedWidth });
            var mean = new[] { 102.9801f, 115.9465f, 122.7717f };
            for (int y = paddedHeight - image.Height; y < image.Height; y++)
            {
                image.ProcessPixelRows(im =>
                {
                    var pixelSpan = im.GetRowSpan(y);
                    for (int x = paddedWidth - image.Width; x < image.Width; x++)
                    {
                        input[0, y, x] = pixelSpan[x].B - mean[0];
                        input[1, y, x] = pixelSpan[x].G - mean[1];
                        input[2, y, x] = pixelSpan[x].R - mean[2];
                    }
                });

            }

            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", input)
            };

            // Run inference
            using var session = new InferenceSession(modelFilePath);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Postprocess to get predictions
            var resultsArray = results.ToArray();
            float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
            long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
            float[] confidences = resultsArray[2].AsEnumerable<float>().ToArray();



            var predictions = new List<Prediction>();
            var minConfidence = 0.7f;
            for (int i = 0; i < boxes.Length - 4; i += 4)
            {
                var index = i / 4;
                if (confidences[index] >= minConfidence)
                {
                    predictions.Add(new Prediction
                    {
                        Box = new Box(boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]),
                        Label = LabelMap.Labels[labels[index]],
                        Confidence = confidences[index]
                    });
                }
            }

            // Put boxes, labels and confidence on image and save for viewing
            using var outputImage = File.OpenWrite(outImageFilePath);
            Font font = SystemFonts.CreateFont("SimHei", 16);
            foreach (var p in predictions)
            {
                image.Mutate(x =>
                {
                    x.DrawLines(Color.Red, 2f, new PointF[] {

                        new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin),
                        new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymin),

                        new PointF(p.Box.Xmax, p.Box.Ymin),
                        new PointF(p.Box.Xmax, p.Box.Ymax),

                        new PointF(p.Box.Xmax, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymax),

                        new PointF(p.Box.Xmin, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymin)
                    });
                    x.DrawText($"{p.Label}, {p.Confidence:0.00}", font, Color.Red, new PointF(p.Box.Xmin, p.Box.Ymin));
                });
            }
            image.SaveAsJpeg(outputImage);
        }
    }
    public class Prediction
    {
        public Box Box { set; get; }
        public string Label { set; get; }
        public float Confidence { set; get; }
    }
    public class Box
    {
        public Box(float xMin, float yMin, float xMax, float yMax)
        {
            Xmin = xMin;
            Ymin = yMin;
            Xmax = xMax;
            Ymax = yMax;
        }
        public float Xmin { set; get; }
        public float Xmax { set; get; }
        public float Ymin { set; get; }
        public float Ymax { set; get; }
    }

    public static class LabelMap
    {
        static LabelMap()
        {
            Labels = new string[]
            {
                "",
                "人",
                "自行车",
                "汽车",
                "摩托车",
                "飞机",
                "总线",
                "火车",
                "卡车",
                "船",
                "红绿灯",
                "消火栓",
                "停止标志",
                "停车收费表",
                "长凳",
                "鸟",
                "猫",
                "狗",
                "马",
                "羊",
                "奶牛",
                "大象",
                "熊",
                "斑马",
                "长颈鹿",
                "背包",
                "雨伞",
                "手提包",
                "领带",
                "手提箱",
                "飞盘",
                "滑雪板",
                "滑雪板",
                "运动球",
                "风筝",
                "棒球棒",
                "棒球手套",
                "滑板",
                "冲浪板",
                "网球拍",
                "瓶子",
                "酒杯",
                "杯子",
                "叉子",
                "刀",
                "勺子",
                "碗",
                "香蕉",
                "苹果",
                "三明治",
                "橙色",
                "西兰花",
                "胡萝卜",
                "热狗",
                "披萨",
                "甜甜圈",
                "蛋糕",
                "椅子",
                "沙发",
                "盆栽植物",
                "床",
                "餐桌",
                "厕所",
                "电视",
                "笔记本电脑",
                "鼠标",
                "远程",
                "键盘",
                "手机",
                "微波",
                "烤箱",
                "烤面包机",
                "水槽",
                "冰箱",
                "书",
                "时钟",
                "花瓶",
                "剪刀",
                "泰迪熊",
                "吹风机",
                "牙刷"

            };
        }


        public static string[] Labels { set; get; }
    }
}