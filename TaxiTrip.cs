﻿using System;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Text;

namespace TaxiFareLearning
{

	/// <summary>
	///Mapping the csv files for the model
	/// </summary>
	class TaxiTrip
	{
		[LoadColumn(0)]
		public string VendorId;

		[LoadColumn(1)]
		public string RateCode;

		[LoadColumn(2)]
		public float PassengerCount;

		[LoadColumn(3)]
		public float TripTime;

		[LoadColumn(4)]
		public float TripDistance;

		[LoadColumn(5)]
		public string PaymentType;

		[LoadColumn(6)]
		public float FareAmount;

	}

	//Prediction column for the program
	public class TaxiTripFarePrediction
	{
		[ColumnName("Score")]
		public float FareAmount;
	}

}
