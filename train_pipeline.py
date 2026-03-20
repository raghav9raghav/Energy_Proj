{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "bc3ca66c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026/03/03 15:28:52 INFO mlflow.tracking.fluent: Experiment with name 'train_pipeline' does not exist. Creating a new experiment.\n",
      "2026/03/03 15:28:55 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.\n",
      "2026/03/03 15:28:55 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Run rf_n50_v1 logged\n",
      "MAE: 856.3237836484246\n",
      "RMSE: 4592.089155103107\n",
      "----------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026/03/03 15:29:08 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.\n",
      "2026/03/03 15:29:08 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Run rf_n100_v1 logged\n",
      "MAE: 819.1338770398012\n",
      "RMSE: 4320.455810995576\n",
      "----------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026/03/03 15:29:25 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.\n",
      "2026/03/03 15:29:25 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Run rf_n200_v1 logged\n",
      "MAE: 812.1305794361538\n",
      "RMSE: 4186.977749405061\n",
      "----------------------------------------\n",
      "All runs completed.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import mlflow\n",
    "import mlflow.sklearn\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.multioutput import MultiOutputRegressor\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
    "\n",
    "\n",
    "# ==========================================\n",
    "# 1️⃣ Setup MLflow (Professional Setup)\n",
    "# ==========================================\n",
    "tracking_path = os.path.abspath(\"./mlruns\")\n",
    "mlflow.set_tracking_uri(f\"file:///{tracking_path}\")\n",
    "mlflow.set_experiment(\"train_pipeline\")\n",
    "\n",
    "\n",
    "# ==========================================\n",
    "# 2️⃣ Load Dataset\n",
    "# ==========================================\n",
    "df = pd.read_csv(\n",
    "    r\"C:\\Users\\Sparsh Sopory\\Desktop\\global_energy_sustainability_1990_2025_top20.csv\"\n",
    ")\n",
    "\n",
    "TARGETS = [\n",
    "    \"renewables_share_energy\",\n",
    "    \"fossil_share_energy\",\n",
    "    \"energy_per_capita\"\n",
    "]\n",
    "\n",
    "df = df.dropna(subset=TARGETS).dropna()\n",
    "\n",
    "features = [col for col in df.columns \n",
    "            if col not in TARGETS + [\"country\", \"year\"]]\n",
    "\n",
    "X = df[features]\n",
    "y = df[TARGETS]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y,\n",
    "    test_size=0.2,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "\n",
    "# ==========================================\n",
    "# 3️⃣ Train + Log Runs Professionally\n",
    "# ==========================================\n",
    "for n in [50, 100, 200]:\n",
    "\n",
    "    with mlflow.start_run(run_name=f\"rf_n{n}_v1\"):\n",
    "\n",
    "        # Structured Tags (Professional)\n",
    "        mlflow.set_tag(\"project\", \"Energy Sustainability Forecasting\")\n",
    "        mlflow.set_tag(\"stage\", \"development\")\n",
    "        mlflow.set_tag(\"model_family\", \"RandomForest\")\n",
    "        mlflow.set_tag(\"dataset_version\", \"1990_2025_top20\")\n",
    "        mlflow.set_tag(\"pipeline\", \"train_pipeline\")\n",
    "\n",
    "        model = MultiOutputRegressor(\n",
    "            RandomForestRegressor(\n",
    "                n_estimators=n,\n",
    "                random_state=42,\n",
    "                n_jobs=-1\n",
    "            )\n",
    "        )\n",
    "\n",
    "        model.fit(X_train, y_train)\n",
    "        y_pred = model.predict(X_test)\n",
    "\n",
    "        mae = mean_absolute_error(y_test, y_pred)\n",
    "        rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "\n",
    "        # Parameters\n",
    "        mlflow.log_param(\"model_type\", \"RandomForest\")\n",
    "        mlflow.log_param(\"n_estimators\", n)\n",
    "        mlflow.log_param(\"random_state\", 42)\n",
    "\n",
    "        # Metrics\n",
    "        mlflow.log_metric(\"MAE\", mae)\n",
    "        mlflow.log_metric(\"RMSE\", rmse)\n",
    "\n",
    "        # Model Artifact\n",
    "        mlflow.sklearn.log_model(model, \"model\")\n",
    "\n",
    "        print(f\"Run rf_n{n}_v1 logged\")\n",
    "        print(\"MAE:\", mae)\n",
    "        print(\"RMSE:\", rmse)\n",
    "        print(\"-\" * 40)\n",
    "\n",
    "print(\"All runs completed.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f0449730",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "myenvnlp",
   "language": "python",
   "name": "myenvnlp"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
