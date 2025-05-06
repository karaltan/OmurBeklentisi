# Life Expectancy Machine Learning Application

This project is a machine learning application using **LinearRegression, DecisionTreeRegressor** designed to predict life expectancy based on various features. The application utilizes a dataset containing relevant information and implements a model to analyze and predict life expectancy values.

## Project Structure

```
life-expectancy-ml-app
├── data
│   └── life-expectancy.csv       # Dataset containing features related to life expectancy
├── src
│   ├── main.py                    # Entry point for the application
│   ├── preprocess.py              # Data preprocessing functions
│   ├── model.py                   # Model training and evaluation
│   └── utils.py                   # Utility functions for visualization and metrics
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation for the project
```

## Dataset

The dataset used in this project is located in the `data/life-expectancy.csv` file. It includes various features that influence life expectancy, such as health indicators, economic factors, and demographic information.

## Setup Instructions

1. Clone the repository:

   ```
   git clone <repository-url>
   cd life-expectancy-ml-app
   ```
2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```
python src/main.py
```

This will load the dataset, preprocess the data, train the model, and evaluate its performance.

## Model Description

The `LifeExpectancyModel` class in `src/model.py` is responsible for training the machine learning model. It includes methods for:

- Training the model on the preprocessed data
- Making predictions based on input features
- Evaluating the model's performance using various metrics

## Contributing

Contributions to improve the model and application are welcome. Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License.
