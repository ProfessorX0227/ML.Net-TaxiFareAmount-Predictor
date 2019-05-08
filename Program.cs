using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TaxiFareLearning
{
	class Program
	{
		// strings for paths to csv data
		static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
		static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
		static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

		//initalize variables in main for training the trainning data path
		static void Main(string[] args)
		{
			Console.WriteLine(Environment.CurrentDirectory);

			MLContext mlContext = new MLContext(seed: 0);

			var model = Train(mlContext, _trainDataPath);

			Evaluate(mlContext, model);

			TestSinglePrediction(mlContext);

		}

		//training model
		public static ITransformer Train(MLContext mlContext, string dataPath)
		{
			//loading data from training csv file
			IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

			//declaring the the label the program wants to predict is Fare Amount
			var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
			
				//transforming all string values to numeric values for data to be processed
				.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
				.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
				.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
				//declaring all the other headers are features that will be used to predict the label
				.Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))

				//choosing the learning algotithm
				///(FastTreeRegressionTrainer) Builds each regression tree in a step-wise fashion. It uses a pre-defined loss function to measure the error in each step and correct for it in the next. The result is a prediction model that is actually an ensemble of weaker prediction models.
				.Append(mlContext.Regression.Trainers.FastTree());

			Console.WriteLine("=============== Create and Train the Model ===============");
			//train model
			var model = pipeline.Fit(dataView);

			Console.WriteLine("=============== End of training ===============");
			Console.WriteLine();
			//save model
			SaveModelAsFile(mlContext, model);
			return model;

		}

		//Evaluates model
		private static void Evaluate(MLContext mlContext, ITransformer model)
		{
			//takes the model created from training
			//Evaluate(mlContext, model);

			//loads the test dataset csv file
			IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

			//transform data in dataset to matach syntaxing from training model
			var predictions = model.Transform(dataView);

			var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

			Console.WriteLine();
			Console.WriteLine($"*************************************************");
			Console.WriteLine($"*       Model quality metrics evaluation         ");
			Console.WriteLine($"*------------------------------------------------");

			//will output success and buffer room
			Console.WriteLine($"*      Success Rate % Score:      {metrics.RSquared:0.##}");
			Console.WriteLine($"*       RMS Room for Error:      {metrics.Rms:#.##}");
			Console.WriteLine($"*************************************************");
		}

		//executes a single test prediction with certain values
		private static void TestSinglePrediction(MLContext mlContext)

		{
			ITransformer loadedModel;
			using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
			{
				loadedModel = mlContext.Model.Load(stream);
			}

			var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);

			var taxiTripSample = new TaxiTrip()
			{
				//VendorId = "VTS",
				//RateCode = "1",
				//PassengerCount = 1,
				//TripTime = 1140,
				//TripDistance = 3.75f,
				//PaymentType = "CRD",
				//FareAmount = 0 // To predict. Actual/Observed = 0

				VendorId = "VTS",
				RateCode = "1",
				PassengerCount = 2,
				TripTime = 1130,
				TripDistance = 4.13f,
				PaymentType = "CRD",
				FareAmount = 0 // To predict. Actual/Observed = 0
			};

			var prediction = predictionFunction.Predict(taxiTripSample);

			//var prediction = predictionFunction.Predict(new TaxiTrip { PassengerCount = 1, PaymentType = "CSH", TripTime = 800, VendorId = "VTS", RateCode = "1", TripDistance = 3.75f });

			Console.WriteLine($"**********************************************************************");
			Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
			Console.WriteLine($"**********************************************************************");
			Console.ReadLine();
		}

		//saves madel as a .zip file
		private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
		{

			//Save the model so that it can be reused and consumed in other applications. The ITransformer has a SaveTo(IHostEnvironment, Stream) method that takes in the _modelPath global field, and a Stream. Since we want to save this as a zip file, we'll create the FileStream immediately before calling the SaveTo method. Add the following code to the SaveModelAsFile method as the next line:
			using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
				mlContext.Model.Save(model, fileStream);

			Console.WriteLine("The model is saved to {0}", _modelPath);
		}

	}
}
