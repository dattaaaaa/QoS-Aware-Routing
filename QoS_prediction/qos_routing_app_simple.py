import sys
import os
import pandas as pd
import numpy as np

# --- PyQt6 Imports for the GUI ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QTableWidget, QTableWidgetItem, QDoubleSpinBox,
    QStatusBar, QMessageBox, QTabWidget, QHeaderView
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont

# --- Matplotlib for plotting within PyQt ---
# CORRECTED: Use the qt6agg backend for PyQt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- PySpark for Backend Processing ---
from pyspark.sql import SparkSession, types as T, functions as F
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
# CORRECTED: Import the necessary function to handle vectors
from pyspark.ml.functions import vector_to_array

# =============================================================================
#  BACKEND: SPARK PROCESSING LOGIC (Corrected Version)
# =============================================================================

class SparkProcessor:
    """Handles all Spark-related data processing."""

    def __init__(self):
        print("Initializing Spark Session...")
        if 'HADOOP_HOME' not in os.environ:
            dummy_hadoop_path = os.path.join(os.getcwd(), 'dummy_hadoop')
            bin_path = os.path.join(dummy_hadoop_path, 'bin')
            if not os.path.exists(bin_path):
                os.makedirs(bin_path)
            if not os.path.exists(os.path.join(bin_path, 'winutils.exe')):
                with open(os.path.join(bin_path, 'winutils.exe'), 'w') as f:
                    f.write('')
            os.environ['HADOOP_HOME'] = dummy_hadoop_path
            
        self.spark = SparkSession.builder \
            .appName("QoSRoutingOptimization") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        print("Spark Session created successfully.")

    def stop_spark(self):
        print("Stopping Spark session.")
        self.spark.stop()

    def process_data(self, filepath, weights):
        """
        The main data processing pipeline.
        Steps: Load -> Preprocess -> Normalize -> Engineer Features -> Score -> Select Best.
        """
        df = self.spark.read.csv(filepath, header=True, inferSchema=True)

        required_cols = [
            'id', 'PCell_RSRP_max', 'PCell_SNR_max', 'speed_kmh',
            'temperature', 'pressure', 'Traffic Jam Factor', 'target'
        ]
        existing_cols = [col for col in required_cols if col in df.columns]
        df_clean = df.select(existing_cols).dropna(subset=existing_cols)

        cols_to_normalize = [
            'PCell_RSRP_max', 'PCell_SNR_max', 'speed_kmh',
            'temperature', 'pressure', 'Traffic Jam Factor'
        ]
        
        stages = []
        for col_name in cols_to_normalize:
            vec_assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vec", handleInvalid="skip")
            scaler = MinMaxScaler(inputCol=f"{col_name}_vec", outputCol=f"{col_name}_norm")
            stages += [vec_assembler, scaler]

        scaler_pipeline = Pipeline(stages=stages)
        scaler_model = scaler_pipeline.fit(df_clean)
        df_scaled = scaler_model.transform(df_clean)
        
        # --- FINAL FIX: Use vector_to_array to properly extract the value ---
        for col_name in cols_to_normalize:
            df_scaled = df_scaled.withColumn(
                f"{col_name}_norm_val",
                # 1. Convert the VectorUDT to a standard ArrayType
                # 2. Get the first element (index 0) from that array
                vector_to_array(F.col(f"{col_name}_norm"))[0]
            )

        # --- Step 3: Feature Engineering ---
        df_featured = df_scaled.withColumn(
            "SignalScore", (0.6 * F.col('PCell_RSRP_max_norm_val')) + (0.4 * F.col('PCell_SNR_max_norm_val'))
        ).withColumn(
            "MobilityScore", 1.0 - F.col('speed_kmh_norm_val')
        ).withColumn(
            "EnvScore", (0.5 * F.col('temperature_norm_val')) + (0.5 * F.col('pressure_norm_val'))
        ).withColumn(
            "CongestionPenalty", F.col('Traffic Jam Factor_norm_val')
        )

        # --- Step 4: Weighted Scoring Algorithm ---
        w1, w2, w3, w4 = weights['w_signal'], weights['w_mobility'], weights['w_env'], weights['w_congestion']
        
        df_scored = df_featured.withColumn(
            "QoS_Score",
            (w1 * F.col("SignalScore")) +
            (w2 * F.col("MobilityScore")) +
            (w3 * F.col("EnvScore")) -
            (w4 * F.col("CongestionPenalty"))
        )

        # --- Step 5: Route Selection & Prepare results ---
        best_route_spark = df_scored.orderBy(F.col("QoS_Score").desc()).first()

        display_cols = [
            'id', 'SignalScore', 'MobilityScore', 'EnvScore', 'CongestionPenalty', 'QoS_Score'
        ]
        
        full_results_pd = df_scored.select(display_cols).limit(1000).toPandas()
        best_route_pd = pd.DataFrame([best_route_spark.asDict()], columns=best_route_spark.asDict().keys())
        best_route_pd = best_route_pd[display_cols]

        return full_results_pd, best_route_pd


# =============================================================================
#  GUI THREADING LOGIC (No changes here)
# =============================================================================
class Worker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    def __init__(self, processor, filepath, weights):
        super().__init__()
        self.processor = processor
        self.filepath = filepath
        self.weights = weights
    def run(self):
        try:
            result = self.processor.process_data(self.filepath, self.weights)
            self.finished.emit(result)
        except Exception as e:
            # Catch PySpark's verbose Java errors
            error_str = str(e)
            if 'Py4JJavaError' in error_str:
                # Extract the core Spark SQL error message which is more readable
                import re
                match = re.search(r":\s(org\.apache\.spark\.SparkSQLException:.*)", error_str, re.DOTALL)
                if match:
                    error_str = match.group(1).split('\n\t at')[0]
            self.error.emit(f"An error occurred during processing:\n{error_str}")

# =============================================================================
#  FRONTEND: PYQT6 GUI APPLICATION (No changes here)
# =============================================================================
class QoSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QoS-Aware Routing Optimization Framework")
        self.setGeometry(100, 100, 1200, 800)
        self.spark_processor = SparkProcessor()
        self.filepath = None
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.init_ui()
    def init_ui(self):
        title_label = QLabel("Big Data-Driven Routing Optimization")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(title_label)
        controls_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Network Data")
        self.load_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file loaded.")
        self.weights_spinners = {}
        weights_layout = QHBoxLayout()
        for key, name, default_val in [('w_signal', 'Signal (w1)', 0.4), ('w_mobility', 'Mobility (w2)', 0.2),('w_env', 'Environment (w3)', 0.1), ('w_congestion', 'Congestion (w4)', 0.3)]:
            weights_layout.addWidget(QLabel(f"{name}:"))
            spinner = QDoubleSpinBox()
            spinner.setRange(0.0, 1.0)
            spinner.setSingleStep(0.05)
            spinner.setValue(default_val)
            self.weights_spinners[key] = spinner
            weights_layout.addWidget(spinner)
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_analysis)
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.file_label, 1)
        controls_layout.addLayout(weights_layout)
        controls_layout.addWidget(self.run_button)
        self.main_layout.addLayout(controls_layout)
        self.tabs = QTabWidget()
        self.full_data_table = QTableWidget()
        self.tabs.addTab(self.full_data_table, "All Route Scores (Sample)")
        self.best_route_table = QTableWidget()
        self.tabs.addTab(self.best_route_table, "Selected Best Route")
        self.main_layout.addWidget(self.tabs, 1)
        self.plot_canvas = FigureCanvas(Figure(figsize=(10, 4)))
        self.main_layout.addWidget(self.plot_canvas)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Please load data.")
    def select_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Network Data File", "", "CSV Files (*.csv)")
        if filepath:
            self.filepath = filepath
            self.file_label.setText(os.path.basename(filepath))
            self.run_button.setEnabled(True)
            self.status_bar.showMessage(f"Loaded: {self.filepath}")
    def run_analysis(self):
        if not self.filepath:
            QMessageBox.warning(self, "Warning", "Please load a data file first.")
            return
        self.run_button.setEnabled(False)
        self.status_bar.showMessage("Processing... This may take a moment.")
        weights = {key: spinner.value() for key, spinner in self.weights_spinners.items()}
        self.thread = QThread()
        self.worker = Worker(self.spark_processor, self.filepath, weights)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
    def on_analysis_complete(self, results):
        full_df, best_route_df = results
        self.populate_table(self.full_data_table, full_df)
        self.populate_table(self.best_route_table, best_route_df)
        self.update_plot(best_route_df)
        self.status_bar.showMessage("Analysis complete. Best route selected.", 5000)
        self.run_button.setEnabled(True)
    def on_analysis_error(self, error_msg):
        QMessageBox.critical(self, "Error", error_msg)
        self.status_bar.showMessage("An error occurred during analysis.", 5000)
        self.run_button.setEnabled(True)
    def populate_table(self, table_widget, df):
        df = df.round(4)
        table_widget.setRowCount(df.shape[0])
        table_widget.setColumnCount(df.shape[1])
        table_widget.setHorizontalHeaderLabels(df.columns)
        for i, row in enumerate(df.itertuples(index=False)):
            for j, value in enumerate(row):
                table_widget.setItem(i, j, QTableWidgetItem(str(value)))
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    def update_plot(self, best_route_df):
        ax = self.plot_canvas.figure.subplots()
        ax.clear()
        scores = best_route_df[['SignalScore', 'MobilityScore', 'EnvScore', 'CongestionPenalty']].iloc[0]
        score_names = scores.index
        score_values = scores.values
        colors = ['green', 'blue', 'purple', 'red']
        bars = ax.bar(score_names, score_values, color=colors)
        ax.set_title(f"Score Components for Best Route (ID: {best_route_df['id'].iloc[0]})")
        ax.set_ylabel("Normalized Score / Penalty")
        ax.set_ylim(0, 1.1)
        ax.bar_label(bars, fmt='%.3f')
        self.plot_canvas.draw()
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit','Are you sure you want to exit?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.spark_processor.stop_spark()
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QoSApp()
    window.show()
    sys.exit(app.exec())