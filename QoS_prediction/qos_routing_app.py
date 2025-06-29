import sys
import os
import pandas as pd
import numpy as np
from itertools import product
import time

# --- PyQt6 Imports for the GUI ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QTableWidget, QTableWidgetItem, QDoubleSpinBox,
    QStatusBar, QMessageBox, QTabWidget, QHeaderView, QProgressBar, QGroupBox,
    QFormLayout, QTextEdit, QCheckBox, QSpinBox, QComboBox, QSplitter,
    QScrollArea, QFrame
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor

# --- Matplotlib for plotting within PyQt ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- PySpark for Backend Processing ---
from pyspark.sql import SparkSession, types as T, functions as F
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array

# =============================================================================
#  STYLING & CONFIGURATION
# =============================================================================

# Define statuses and their visual properties (Symbol, Text, Color)
STATUS_CONFIG = {
    "PENDING": ("⚪", "Pending", "#888888"),
    "RUNNING": ("🔄", "Running", "#FF8C00"),  # Darker orange
    "SUCCESS": ("✅", "Success", "#00A36C"),  # Green
    "ERROR": ("❌", "Failed",  "#D82E2E"),   # Red
    "OPTIMIZING": ("🔍", "Optimizing", "#9B59B6"),  # Purple
    "COMPLETED": ("🎯", "Completed", "#00A36C"),  # Green
}

APP_STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #2E2E2E;
        color: #E0E0E0;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    QGroupBox {
        font-weight: bold;
        border: 2px solid #555;
        border-radius: 8px;
        margin-top: 1ex;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 5px;
        color: #FFD700;
    }
    QPushButton {
        background-color: #4A90E2;
        color: #FFF;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        border: none;
    }
    QPushButton:hover {
        background-color: #357ABD;
    }
    QPushButton:pressed {
        background-color: #2E5984;
    }
    QPushButton:disabled {
        background-color: #444;
        color: #888;
    }
    QPushButton#optimizeButton {
        background-color: #9B59B6;
    }
    QPushButton#optimizeButton:hover {
        background-color: #8E44AD;
    }
    QPushButton#runButton {
        background-color: #00A36C;
        font-size: 14px;
        padding: 10px 20px;
    }
    QPushButton#runButton:hover {
        background-color: #008A5B;
    }
    QDoubleSpinBox, QSpinBox, QComboBox, QTableWidget {
        background-color: #3C3C3C;
        color: #E0E0E0;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 4px;
    }
    QHeaderView::section {
        background-color: #4C4C4C;
        font-weight: bold;
        padding: 8px;
        border: 1px solid #666;
    }
    QProgressBar {
        border: 2px solid #555;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        background-color: #3C3C3C;
    }
    QProgressBar::chunk {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                        stop:0 #00A36C, stop:1 #4CAF50);
        border-radius: 6px;
    }
    QTextEdit {
        background-color: #1E1E1E;
        color: #E0E0E0;
        border: 1px solid #555;
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 11px;
    }
    QTabWidget::pane {
        border: 1px solid #555;
        border-radius: 4px;
    }
    QTabBar::tab {
        background-color: #444;
        color: #E0E0E0;
        padding: 8px 16px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
    }
    QScrollArea {
        border: none;
        background-color: transparent;
    }
    QFrame {
        border: 1px solid #555;
        border-radius: 4px;
    }
"""

# =============================================================================
#  OPTIMIZATION ALGORITHMS
# =============================================================================

class WeightOptimizer:
    """Various optimization strategies for finding the best weights."""
    
    @staticmethod
    def grid_search(df, target_col='target', n_points=5):
        """Grid search optimization."""
        weight_values = np.linspace(0.1, 0.9, n_points)
        best_weights = None
        best_score = -float('inf')
        total_combinations = n_points ** 4
        tested = 0
        
        # Add this safety check
        if 'SignalScore' not in df.columns or 'MobilityScore' not in df.columns or \
           'EnvScore' not in df.columns or 'CongestionPenalty' not in df.columns:
            yield 0, total_combinations, best_weights, best_score, []
            return
        
        results = []
        
        for w1, w2, w3, w4 in product(weight_values, repeat=4):
            # Add guard against division by zero
            total = w1 + w2 + w3 + w4
            if total < 1e-6:  # Prevent division by near-zero
                w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
            else:
                w1, w2, w3, w4 = w1/total, w2/total, w3/total, w4/total
            
            # Calculate QoS scores
            qos_scores = (w1 * df['SignalScore'] + w2 * df['MobilityScore'] + 
                         w3 * df['EnvScore'] - w4 * df['CongestionPenalty'])
            
            # Correlation with target (higher is better)
            try:
                # Handle cases with constant values
                if qos_scores.nunique() < 2 or df[target_col].nunique() < 2:
                    correlation = 0.0
                else:
                    correlation = qos_scores.corr(df[target_col])
            except Exception as e:
                correlation = 0.0
            
            if correlation > best_score:
                best_score = correlation
                best_weights = {'w_signal': w1, 'w_mobility': w2, 'w_env': w3, 'w_congestion': w4}
            
            results.append({
                'weights': [w1, w2, w3, w4],
                'correlation': correlation
            })
            
            tested += 1
            yield tested, total_combinations, best_weights, best_score, results
    
    @staticmethod
    def random_search(df, target_col='target', n_iterations=100):
        """Random search optimization."""
        best_weights = None
        best_score = -float('inf')
        
        # Add this safety check
        if 'SignalScore' not in df.columns or 'MobilityScore' not in df.columns or \
           'EnvScore' not in df.columns or 'CongestionPenalty' not in df.columns:
            yield 0, n_iterations, best_weights, best_score, []
            return
        
        results = []
        
        for i in range(n_iterations):
            # Generate random weights and normalize
            weights = np.random.random(4)
            total = weights.sum()
            if total < 1e-6:
                weights = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                weights = weights / total
            w1, w2, w3, w4 = weights
            
            # Calculate QoS scores
            qos_scores = (w1 * df['SignalScore'] + w2 * df['MobilityScore'] + 
                         w3 * df['EnvScore'] - w4 * df['CongestionPenalty'])
            
            # Correlation with target
            try:
                # Handle cases with constant values
                if qos_scores.nunique() < 2 or df[target_col].nunique() < 2:
                    correlation = 0.0
                else:
                    correlation = qos_scores.corr(df[target_col])
            except Exception as e:
                correlation = 0.0
            
            if correlation > best_score:
                best_score = correlation
                best_weights = {'w_signal': w1, 'w_mobility': w2, 'w_env': w3, 'w_congestion': w4}
            
            results.append({
                'weights': [w1, w2, w3, w4],
                'correlation': correlation
            })
            
            yield i + 1, n_iterations, best_weights, best_score, results

# =============================================================================
#  BACKEND: ENHANCED SPARK PROCESSING
# =============================================================================

class EnhancedSparkProcessor(QObject):
    """Enhanced Spark processor with optimization capabilities."""
    step_update = pyqtSignal(str, str, str)  # step_key, status, message
    log_message = pyqtSignal(str)  # For detailed logging
    optimization_progress = pyqtSignal(int, int, object, float, object)  # Add this line

    def __init__(self):
        super().__init__()
        self.spark = None

    def start_spark(self):
        if 'HADOOP_HOME' not in os.environ:
            os.environ['HADOOP_HOME'] = os.path.join(os.getcwd(), 'dummy_hadoop')
        self.spark = SparkSession.builder \
            .appName("EnhancedQoSRoutingOptimization") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()

    def stop_spark(self):
        if self.spark:
            self.spark.stop()
            self.spark = None

    def preprocess_and_engineer_features(self, filepath):
        """Load data and engineer features."""
        try:
            # Step 1: Data Ingestion
            self.step_update.emit("ingest", "RUNNING", "Loading data from file...")
            self.log_message.emit(f"Loading data from: {filepath}")
            
            df = self.spark.read.csv(filepath, header=True, inferSchema=True)
            row_count = df.count()
            self.step_update.emit("ingest", "SUCCESS", f"Loaded {row_count} rows.")
            self.log_message.emit(f"Successfully loaded {row_count} rows from CSV")

            # Step 2: Preprocessing
            self.step_update.emit("preprocess", "RUNNING", "Cleaning and filtering data...")
            
            required_cols = ['id', 'PCell_RSRP_max', 'PCell_SNR_max', 'speed_kmh',
                           'temperature', 'pressure', 'Traffic Jam Factor', 'target']
            
            # Check which columns exist
            available_cols = df.columns
            missing_cols = [col for col in required_cols if col not in available_cols]
            
            if missing_cols:
                self.log_message.emit(f"Missing columns: {missing_cols}")
                # Create dummy columns for missing ones
                for col in missing_cols:
                   df = df.withColumn(col, F.lit(0.0)) 
                # Create dummy target if missing
                if 'target' in missing_cols:
                    df = df.withColumn('target', F.lit(1))
                    self.log_message.emit("Created dummy target column")
            
            df_clean = df.select([col for col in required_cols if col in available_cols]).dropna()
            clean_count = df_clean.count()
            self.step_update.emit("preprocess", "SUCCESS", f"Valid rows after cleaning: {clean_count}")
            self.log_message.emit(f"Data cleaned: {clean_count} valid rows remaining")

            # Step 3: Feature Engineering
            self.step_update.emit("feature_eng", "RUNNING", "Normalizing features and creating scores...")
            
            cols_to_normalize = ['PCell_RSRP_max', 'PCell_SNR_max', 'speed_kmh',
                               'temperature', 'pressure', 'Traffic Jam Factor']
            
            # Build normalization pipeline
            stages = []
            for col in cols_to_normalize:
                if col in df_clean.columns:
                    stages.append(VectorAssembler(inputCols=[col], outputCol=f"{col}_vec"))
                    stages.append(MinMaxScaler(inputCol=f"{col}_vec", outputCol=f"{col}_norm"))

            scaler_pipeline = Pipeline(stages=stages).fit(df_clean)
            df_scaled = scaler_pipeline.transform(df_clean)
            
            # Extract normalized values
            for col in cols_to_normalize:
                if col in df_clean.columns:
                    df_scaled = df_scaled.withColumn(f"{col}_norm_val", 
                                                   vector_to_array(F.col(f"{col}_norm"))[0])

            # Calculate composite scores
            df_featured = df_scaled.withColumn(
                "SignalScore", 
                F.when(F.col('PCell_RSRP_max_norm_val').isNotNull() & F.col('PCell_SNR_max_norm_val').isNotNull(),
                       0.6 * F.col('PCell_RSRP_max_norm_val') + 0.4 * F.col('PCell_SNR_max_norm_val'))
                .otherwise(0.0)
            ).withColumn(
                "MobilityScore", 
                F.when(F.col('speed_kmh_norm_val').isNotNull(),
                       1.0 - F.col('speed_kmh_norm_val'))
                .otherwise(0.0)
            ).withColumn(
                "EnvScore", 
                F.when(F.col('temperature_norm_val').isNotNull() & F.col('pressure_norm_val').isNotNull(),
                       0.5 * F.col('temperature_norm_val') + 0.5 * F.col('pressure_norm_val'))
                .otherwise(0.0)
            ).withColumn(
                "CongestionPenalty", 
                F.when(F.col('Traffic Jam Factor_norm_val').isNotNull(),
                       F.col('Traffic Jam Factor_norm_val'))
                .otherwise(0.0)
            )
            
            self.step_update.emit("feature_eng", "SUCCESS", "Feature engineering completed.")
            self.log_message.emit("All feature scores computed successfully")

            # Convert to pandas for optimization
            feature_cols = ['id', 'SignalScore', 'MobilityScore', 'EnvScore', 'CongestionPenalty', 'target']
            df_pandas = df_featured.select(feature_cols).toPandas()
            
            return df_pandas, df_featured

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log_message.emit(f"Error in preprocessing: {error_details}")
            raise Exception(f"Error in data preprocessing:\n{error_details}")

    def optimize_weights(self, df_pandas, method='grid_search', **kwargs):
        """Optimize weights using specified method."""
        try:
            self.step_update.emit("optimize", "RUNNING", f"Starting {method} optimization...")
            self.log_message.emit(f"Beginning weight optimization using {method}")
            
            optimizer = WeightOptimizer()
            
            if method == 'grid_search':
                optimization_generator = optimizer.grid_search(df_pandas, **kwargs)
            elif method == 'random_search':
                optimization_generator = optimizer.random_search(df_pandas, **kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            best_weights = None
            best_score = -float('inf')
            all_results = []
            
            try:
                for current, total, weights, score, results in optimization_generator:
                    if weights:
                        best_weights = weights
                        best_score = score
                        all_results = results
                    
                    self.optimization_progress.emit(current, total, weights, score, results)
                    
                    # Allow GUI to update
                    QApplication.processEvents()
            except StopIteration:
                # Handle normal completion of generator
                pass
            except Exception as e:
                self.log_message.emit(f"Optimization error: {str(e)}")
                raise
            
            self.step_update.emit("optimize", "SUCCESS", 
                                f"Optimization complete. Best correlation: {best_score:.4f}")
            self.log_message.emit(f"Weight optimization completed with correlation: {best_score:.4f}")
            
            return best_weights, best_score, all_results
            
        except Exception as e:
            self.log_message.emit(f"Error in optimization: {str(e)}")
            raise

    def calculate_final_scores(self, df_spark, weights):
        """Calculate final QoS scores and select best route."""
        try:
            self.step_update.emit("scoring", "RUNNING", "Applying optimized weights...")
            
            w1, w2, w3, w4 = weights['w_signal'], weights['w_mobility'], weights['w_env'], weights['w_congestion']
            
            df_scored = df_spark.withColumn(
                "QoS_Score",
                (w1 * F.col("SignalScore")) + (w2 * F.col("MobilityScore")) +
                (w3 * F.col("EnvScore")) - (w4 * F.col("CongestionPenalty"))
            )
            
            self.step_update.emit("scoring", "SUCCESS", "QoS scores calculated with optimized weights.")
            self.log_message.emit("Applied optimized weights to calculate final QoS scores")

            # Step 6: Route Selection
            self.step_update.emit("selection", "RUNNING", "Selecting best route...")
            
            best_route_spark = df_scored.orderBy(F.col("QoS_Score").desc()).first()
            
            self.step_update.emit("selection", "SUCCESS", 
                                f"Best route selected (ID: {best_route_spark['id']})")
            self.log_message.emit(f"Selected best route with ID: {best_route_spark['id']}")

            # Prepare results
            display_cols = ['id', 'SignalScore', 'MobilityScore', 'EnvScore', 'CongestionPenalty', 'QoS_Score']
            full_results_pd = df_scored.select(display_cols).limit(1000).toPandas()
            best_route_pd = pd.DataFrame([best_route_spark.asDict()])[display_cols]

            return full_results_pd, best_route_pd

        except Exception as e:
            self.log_message.emit(f"Error in final scoring: {str(e)}")
            raise

# =============================================================================
#  GUI WIDGETS
# =============================================================================

class AnimatedStatusLabel(QFrame):
    """Enhanced status widget with animations and better visuals."""
    def __init__(self, label_text):
        super().__init__()
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(8, 4, 8, 4)

        self.label = QLabel(label_text)
        self.label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.label.setMinimumWidth(150)
        
        self.status_indicator = QLabel()
        self.status_indicator.setFixedWidth(100)
        self.status_indicator.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        
        self.status_message = QLabel()
        self.status_message.setFont(QFont("Segoe UI", 9))
        
        # Add a subtle frame
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("QFrame { background-color: #3A3A3A; border-radius: 4px; }")
        
        self.main_layout.addWidget(self.label)
        self.main_layout.addStretch()
        self.main_layout.addWidget(self.status_indicator)
        self.main_layout.addWidget(self.status_message, 1)
        
        self.set_status("PENDING")

    def set_status(self, status, message=""):
        symbol, text, color = STATUS_CONFIG[status]
        self.status_indicator.setText(f"{symbol} {text}")
        self.status_indicator.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.status_message.setText(message)

class OptimizationProgressWidget(QWidget):
    """Widget to show optimization progress with real-time updates."""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Progress info
        self.progress_label = QLabel("Optimization Progress")
        self.progress_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        
        self.best_weights_label = QLabel("Best weights will appear here...")
        self.best_score_label = QLabel("Best correlation: N/A")
        
        # Canvas for real-time plotting
        self.figure = Figure(figsize=(8, 4), facecolor='#2E2E2E')
        self.canvas = FigureCanvas(self.figure)
        
        self.layout.addWidget(self.progress_label)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.best_weights_label)
        self.layout.addWidget(self.best_score_label)
        self.layout.addWidget(self.canvas)
        
        self.correlations = []

    def update_progress(self, current, total, best_weights, best_score, results):
        # Handle None values safely
        if best_weights is None:
            best_weights = {}
        if results is None:
            results = []
        if not isinstance(results, list):
            results = []
        if not isinstance(best_weights, dict):
            best_weights = {}
        
        # Update progress bar
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total} ({current/total*100:.1f}%)")
        
        # Update best weights
        if best_weights:
            weights_text = (f"Signal: {best_weights.get('w_signal', 0):.3f}, "
                        f"Mobility: {best_weights.get('w_mobility', 0):.3f}, "
                        f"Environment: {best_weights.get('w_env', 0):.3f}, "
                        f"Congestion: {best_weights.get('w_congestion', 0):.3f}")
            self.best_weights_label.setText(f"Best weights: {weights_text}")
            
        # Update score
        self.best_score_label.setText(f"Best correlation: {best_score:.4f}")
        
        # Update plot - add try-except block
        try:
            if results:
                self.correlations = [r.get('correlation', 0) if isinstance(r, dict) else 0 for r in results]
                self.update_plot()
        except Exception as e:
            print(f"Plot update error: {e}")

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor='#3C3C3C')
        
        if self.correlations:
            ax.plot(self.correlations, color='#4A90E2', linewidth=2)
            ax.fill_between(range(len(self.correlations)), self.correlations, alpha=0.3, color='#4A90E2')
            
            # Highlight best score
            best_idx = np.argmax(self.correlations)
            ax.scatter([best_idx], [max(self.correlations)], color='#FFD700', s=100, zorder=5)
            
        ax.set_title("Optimization Progress: Correlation with Target", color='white', fontweight='bold')
        ax.set_xlabel("Iteration", color='white')
        ax.set_ylabel("Correlation", color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()

class LogWidget(QTextEdit):
    """Enhanced logging widget with timestamps."""
    def __init__(self):
        super().__init__()
        self.setMaximumHeight(150)
        
    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.append(formatted_message)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

# =============================================================================
#  WORKER THREAD
# =============================================================================

class EnhancedWorker(QObject):
    """Enhanced worker for processing with optimization."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    #step_update = pyqtSignal(str, str, str)
    #optimization_progress = pyqtSignal(int, int, object, float, object)
    #log_message = pyqtSignal(str)

    def __init__(self, processor, filepath, weights=None, auto_optimize=False, opt_method='grid_search', opt_params=None):
        super().__init__()
        self.processor = processor
        self.filepath = filepath
        self.weights = weights
        self.auto_optimize = auto_optimize
        self.opt_method = opt_method
        self.opt_params = opt_params or {}
        
        # Connect signals - remove the optimization_progress connection
        #self.processor.step_update.connect(self.step_update.emit)
        #self.processor.log_message.connect(self.log_message.emit)
        #self.processor.optimization_progress.connect(self.optimization_progress.emit)


    def run(self):
        try:
            self.processor.start_spark()
            
            # Preprocess and engineer features
            df_pandas, df_spark = self.processor.preprocess_and_engineer_features(self.filepath)
            
            final_weights = self.weights
            optimization_results = None
            
            if self.auto_optimize:
                # Optimize weights
                best_weights, best_score, opt_results = self.processor.optimize_weights(
                    df_pandas, self.opt_method, **self.opt_params)
                final_weights = best_weights
                optimization_results = {
                    'best_score': best_score,
                    'results': opt_results,
                    'method': self.opt_method
                }
            
            # Calculate final scores
            full_results, best_route = self.processor.calculate_final_scores(df_spark, final_weights)
            
            result = {
                'full_results': full_results,
                'best_route': best_route,
                'final_weights': final_weights,
                'optimization_results': optimization_results
            }
            
            self.finished.emit(result)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log_message.emit(f"CRITICAL ERROR: {error_details}")
            self.error.emit(f"Analysis failed: {str(e)}\n\nDetails: {error_details}")
        finally:
            try:
                self.processor.stop_spark()
            except:
                pass

# =============================================================================
#  MAIN APPLICATION WINDOW
# =============================================================================

class EnhancedQoSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced QoS-Aware Routing Optimization Framework")
        self.setGeometry(50, 50, 1600, 1000)
        self.setStyleSheet(APP_STYLESHEET)
        
        self.filepath = None
        self.steps = ["ingest", "preprocess", "feature_eng", "optimize", "scoring", "selection"]
        
        self.init_ui()
        
    def init_ui(self):
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # Left panel (controls and process flow)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Configuration group
        config_group = QGroupBox("📁 Configuration")
        config_layout = QFormLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.load_button = QPushButton("📂 Load Data File")
        self.load_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #FFD700;")
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.file_label)
        config_layout.addRow(file_layout)
        
        # Optimization settings
        self.auto_optimize_checkbox = QCheckBox("Auto-optimize weights")
        self.auto_optimize_checkbox.setChecked(True)
        self.auto_optimize_checkbox.toggled.connect(self.toggle_manual_weights)
        config_layout.addRow(self.auto_optimize_checkbox)
        
        self.opt_method_combo = QComboBox()
        self.opt_method_combo.addItems(["grid_search", "random_search"])
        config_layout.addRow("Optimization method:", self.opt_method_combo)
        
        self.opt_iterations_spin = QSpinBox()
        self.opt_iterations_spin.setRange(10, 1000)
        self.opt_iterations_spin.setValue(100)
        config_layout.addRow("Iterations (random search):", self.opt_iterations_spin)
        
        self.grid_points_spin = QSpinBox()
        self.grid_points_spin.setRange(3, 10)
        self.grid_points_spin.setValue(5)
        config_layout.addRow("Grid points (grid search):", self.grid_points_spin)
        
        config_group.setLayout(config_layout)
        
        # Manual weights group
        self.weights_group = QGroupBox("⚖️ Manual Weights")
        weights_layout = QFormLayout()
        
        self.weights_spinners = {}
        for key, name, default_val in [
            ('w_signal', 'Signal Weight', 0.4), 
            ('w_mobility', 'Mobility Weight', 0.2),
            ('w_env', 'Environment Weight', 0.1), 
            ('w_congestion', 'Congestion Weight', 0.3)
        ]:
            spinner = QDoubleSpinBox()
            spinner.setRange(0.0, 1.0)
            spinner.setSingleStep(0.01)
            spinner.setValue(default_val)
            spinner.setDecimals(3)
            self.weights_spinners[key] = spinner
            weights_layout.addRow(QLabel(name), spinner)
        
        # Normalize button
        normalize_btn = QPushButton("🔄 Normalize Weights")
        normalize_btn.clicked.connect(self.normalize_weights)
        weights_layout.addRow(normalize_btn)
        
        self.weights_group.setLayout(weights_layout)
        
        # Process flow group
        process_group = QGroupBox("🔄 Process Flow")
        process_layout = QVBoxLayout()
        
        self.status_labels = {}
        step_names = {
            "ingest": "Data Ingestion",
            "preprocess": "Data Preprocessing", 
            "feature_eng": "Feature Engineering",
            "optimize": "Weight Optimization",
            "scoring": "QoS Scoring",
            "selection": "Route Selection"
        }
        
        for step_key in self.steps:
            status_label = AnimatedStatusLabel(step_names[step_key])
            self.status_labels[step_key] = status_label
            process_layout.addWidget(status_label)
        
        self.main_progress_bar = QProgressBar()
        self.main_progress_bar.setRange(0, len(self.steps))
        process_layout.addWidget(self.main_progress_bar)
        
        process_group.setLayout(process_layout)
        
        # Run button
        self.run_button = QPushButton("🚀 RUN ANALYSIS")
        self.run_button.setObjectName("runButton")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_analysis)
        
        # Add to left layout
        left_layout.addWidget(config_group)
        left_layout.addWidget(self.weights_group)
        left_layout.addWidget(process_group)
        left_layout.addWidget(self.run_button)
        left_layout.addStretch()
        
        # Right panel (results and optimization)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Optimization progress
        self.opt_progress_widget = OptimizationProgressWidget()
        self.opt_progress_widget.setVisible(False)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Results table tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        self.results_table = QTableWidget()
        self.results_table.setSortingEnabled(True)
        results_layout.addWidget(QLabel("📊 Route Analysis Results"))
        results_layout.addWidget(self.results_table)
        
        self.results_tabs.addTab(results_widget, "📈 All Routes")
        
        # Best route tab
        best_route_widget = QWidget()
        best_route_layout = QVBoxLayout(best_route_widget)
        
        self.best_route_table = QTableWidget()
        best_route_layout.addWidget(QLabel("🎯 Selected Best Route"))
        best_route_layout.addWidget(self.best_route_table)
        
        self.results_tabs.addTab(best_route_widget, "🏆 Best Route")
        
        # Visualization tab
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        self.viz_canvas = FigureCanvas(Figure(figsize=(12, 8), facecolor='#2E2E2E'))
        viz_layout.addWidget(QLabel("📊 Score Visualization"))
        viz_layout.addWidget(self.viz_canvas)
        
        self.results_tabs.addTab(viz_widget, "📊 Visualization")
        
        # Weights comparison tab
        weights_viz_widget = QWidget()
        weights_viz_layout = QVBoxLayout(weights_viz_widget)
        
        self.weights_canvas = FigureCanvas(Figure(figsize=(10, 6), facecolor='#2E2E2E'))
        weights_viz_layout.addWidget(QLabel("⚖️ Weights Analysis"))
        weights_viz_layout.addWidget(self.weights_canvas)
        
        self.results_tabs.addTab(weights_viz_widget, "⚖️ Weights")
        
        # Log tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        self.log_widget = LogWidget()
        log_layout.addWidget(QLabel("📝 Process Log"))
        log_layout.addWidget(self.log_widget)
        
        self.results_tabs.addTab(log_widget, "📝 Logs")
        
        # Add to right layout
        right_layout.addWidget(self.opt_progress_widget)
        right_layout.addWidget(self.results_tabs)
        
        # Add panels to splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([400, 1200])
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("🚀 Ready! Please load a data file to begin analysis.")
        
        # Initialize UI state
        self.toggle_manual_weights()

    def toggle_manual_weights(self):
        """Toggle manual weights visibility based on auto-optimize checkbox."""
        is_manual = not self.auto_optimize_checkbox.isChecked()
        self.weights_group.setVisible(is_manual)
        
        if self.auto_optimize_checkbox.isChecked():
            self.status_labels["optimize"].setVisible(True)
        else:
            self.status_labels["optimize"].setVisible(False)

    def normalize_weights(self):
        """Normalize manual weights to sum to 1."""
        total = sum(spinner.value() for spinner in self.weights_spinners.values())
        if total > 0:
            for spinner in self.weights_spinners.values():
                spinner.setValue(spinner.value() / total)

    def reset_ui_for_run(self):
        """Reset UI elements before starting a new run."""
        for step in self.steps:
            self.status_labels[step].set_status("PENDING")
        
        self.main_progress_bar.setValue(0)
        self.opt_progress_widget.setVisible(False)
        
        # Clear tables
        self.results_table.setRowCount(0)
        self.best_route_table.setRowCount(0)
        
        # Clear plots
        self.viz_canvas.figure.clear()
        self.viz_canvas.draw()
        self.weights_canvas.figure.clear()
        self.weights_canvas.draw()
        
        # Clear log
        self.log_widget.clear()
        self.log_widget.log_message("Starting new analysis...")

    def select_file(self):
        """Open file dialog to select data file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Network Data File", "", "CSV Files (*.csv);;All Files (*)")
        
        if filepath:
            self.filepath = filepath
            filename = os.path.basename(filepath)
            self.file_label.setText(f"✅ {filename}")
            self.run_button.setEnabled(True)
            self.status_bar.showMessage(f"📁 Data file loaded: {filename}")
            self.log_widget.log_message(f"Loaded data file: {filepath}")

    def run_analysis(self):
        """Start the analysis process."""
        self.reset_ui_for_run()
        self.run_button.setEnabled(False)
        self.status_bar.showMessage("🔄 Analysis in progress...")
        
        # Get configuration
        auto_optimize = self.auto_optimize_checkbox.isChecked()
        opt_method = self.opt_method_combo.currentText()
        
        # Prepare optimization parameters
        opt_params = {}
        if opt_method == 'random_search':
            opt_params['n_iterations'] = self.opt_iterations_spin.value()
        elif opt_method == 'grid_search':
            opt_params['n_points'] = self.grid_points_spin.value()
        
        # Get manual weights if not auto-optimizing
        manual_weights = None
        if not auto_optimize:
            manual_weights = {key: spinner.value() for key, spinner in self.weights_spinners.items()}
        
        # Show optimization progress if auto-optimizing
        if auto_optimize:
            self.opt_progress_widget.setVisible(True)
        
        # Create and start worker thread
        self.thread = QThread()
        self.processor = EnhancedSparkProcessor()
        self.worker = EnhancedWorker(
            self.processor, self.filepath, manual_weights, 
            auto_optimize, opt_method, opt_params)
        
        self.worker.moveToThread(self.thread)
        
        # Connect signals - DO NOT connect worker.step_update
        # Instead, connect processor signals directly to UI
        self.processor.step_update.connect(self.update_step_status)
        self.processor.log_message.connect(self.log_widget.log_message)
        self.processor.optimization_progress.connect(self.opt_progress_widget.update_progress)
        
        # Connect worker signals
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        
        # Cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def update_step_status(self, step_key, status, message):
        """Update individual step status."""
        if step_key in self.status_labels:
            self.status_labels[step_key].set_status(status, message)
            
            if status == "SUCCESS":
                current_value = self.main_progress_bar.value()
                self.main_progress_bar.setValue(current_value + 1)
            elif status == "ERROR":
                self.main_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #D82E2E; }")



    def on_analysis_complete(self, results):
        """Handle completion of analysis."""
        try:
            full_results = results['full_results']
            best_route = results['best_route']
            final_weights = results['final_weights']
            opt_results = results['optimization_results']
            
            # Populate tables
            self.populate_table(self.results_table, full_results)
            self.populate_table(self.best_route_table, best_route)
            
            # Create visualizations
            self.create_score_visualization(full_results, best_route)
            self.create_weights_visualization(final_weights, opt_results)
            
            # Update status
            best_route_id = best_route['id'].iloc[0] if not best_route.empty else "Unknown"
            final_score = best_route['QoS_Score'].iloc[0] if not best_route.empty else 0
            
            self.status_bar.showMessage(
                f"✅ Analysis complete! Best route: {best_route_id} (Score: {final_score:.4f})", 15000)
            
            self.log_widget.log_message(f"Analysis completed successfully")
            self.log_widget.log_message(f"Best route ID: {best_route_id} with QoS score: {final_score:.4f}")
            
            if opt_results:
                self.log_widget.log_message(
                    f"Weight optimization achieved correlation: {opt_results['best_score']:.4f}")
            
        except Exception as e:
            self.log_widget.log_message(f"Error processing results: {str(e)}")
            QMessageBox.warning(self, "Results Error", f"Error processing results:\n{str(e)}")
        
        finally:
            self.run_button.setEnabled(True)

    def on_analysis_error(self, error_msg):
        """Handle analysis errors."""
        # Find the currently running step and mark it as failed
        for step in self.steps:
            status_text = self.status_labels[step].status_indicator.text()
            if "Running" in status_text or "Optimizing" in status_text:
                self.status_labels[step].set_status("ERROR", "Process failed")
                break
        
        self.log_widget.log_message(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n\n{error_msg}")
        self.status_bar.showMessage("❌ Analysis failed. Check logs for details.", 10000)
        self.run_button.setEnabled(True)

    def populate_table(self, table_widget, df):
        """Populate a table widget with DataFrame data."""
        if df.empty:
            return
        
        df_display = df.round(4)
        table_widget.setRowCount(df_display.shape[0])
        table_widget.setColumnCount(df_display.shape[1])
        table_widget.setHorizontalHeaderLabels(df_display.columns.tolist())
        
        for i, row in enumerate(df_display.itertuples(index=False)):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                # Highlight best scores
                if j == df_display.columns.get_loc('QoS_Score') and i == 0:
                    item.setBackground(QColor("#4CAF50"))
                table_widget.setItem(i, j, item)
        
        # Auto-resize columns
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def create_score_visualization(self, full_results, best_route):
        """Create comprehensive score visualization."""
        fig = self.viz_canvas.figure
        fig.clear()
        
        # Create subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Score distribution histogram
        ax1 = fig.add_subplot(gs[0, 0], facecolor='#3C3C3C')
        if not full_results.empty and 'QoS_Score' in full_results.columns:
            ax1.hist(full_results['QoS_Score'], bins=30, alpha=0.7, color='#4A90E2', edgecolor='white')
            if not best_route.empty and 'QoS_Score' in best_route.columns:
                best_score = best_route['QoS_Score'].iloc[0]
            else:
                best_score = 0
            ax1.axvline(best_score, color='#FFD700', linestyle='--', linewidth=3, label=f'Best: {best_score:.4f}')
            ax1.legend()
        ax1.set_title('QoS Score Distribution', color='white', fontweight='bold')
        ax1.set_xlabel('QoS Score', color='white')
        ax1.set_ylabel('Frequency', color='white')
        ax1.tick_params(colors='white')
        
        # 2. Component scores for best route
        ax2 = fig.add_subplot(gs[0, 1], facecolor='#3C3C3C')
        if not best_route.empty:
            components = ['SignalScore', 'MobilityScore', 'EnvScore', 'CongestionPenalty']
            available_components = [c for c in components if c in best_route.columns]
            if available_components:
                values = [best_route[c].iloc[0] for c in available_components]
                colors = ['#00A36C', '#3498DB', '#9B59B6', '#E74C3C']
                bars = ax2.bar(available_components, values, color=colors[:len(available_components)])
                ax2.bar_label(bars, fmt='%.3f', color='white', fontweight='bold')
        ax2.set_title('Best Route Component Scores', color='white', fontweight='bold')
        ax2.set_ylabel('Score', color='white')
        ax2.tick_params(colors='white', axis='x', rotation=45)
        
        # 3. Top 10 routes comparison
        ax3 = fig.add_subplot(gs[1, :], facecolor='#3C3C3C')
        if not full_results.empty and len(full_results) > 0:
            top_10 = full_results.nlargest(10, 'QoS_Score') if 'QoS_Score' in full_results.columns else full_results.head(10)
            if not top_10.empty:
                route_labels = [f"Route {id_val}" for id_val in top_10['id']] if 'id' in top_10.columns else [f"Route {i}" for i in range(len(top_10))]
                qos_scores = top_10['QoS_Score'] if 'QoS_Score' in top_10.columns else [0] * len(top_10)
                
                bars = ax3.bar(range(len(route_labels)), qos_scores, 
                              color=['#FFD700' if i == 0 else '#4A90E2' for i in range(len(route_labels))])
                ax3.set_xticks(range(len(route_labels)))
                ax3.set_xticklabels(route_labels, rotation=45, ha='right')
                ax3.bar_label(bars, fmt='%.3f', color='white', fontweight='bold')
        
        ax3.set_title('Top 10 Routes Comparison', color='white', fontweight='bold')
        ax3.set_ylabel('QoS Score', color='white')
        ax3.tick_params(colors='white')
        
        # Style all axes
        for ax in [ax1, ax2, ax3]:
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.grid(True, alpha=0.3)
        
        self.viz_canvas.draw()

    def create_weights_visualization(self, final_weights, opt_results):
        """Create weights analysis visualization."""
        fig = self.weights_canvas.figure
        fig.clear()
        
        if not final_weights:
            return
        
        # Create subplots
        if opt_results:
            gs = fig.add_gridspec(1, 2, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0], facecolor='#3C3C3C')
            ax2 = fig.add_subplot(gs[0, 1], facecolor='#3C3C3C')
        else:
            ax1 = fig.add_subplot(111, facecolor='#3C3C3C')
        
        # 1. Final weights pie chart
        weight_names = ['Signal', 'Mobility', 'Environment', 'Congestion']
        weight_values = [final_weights['w_signal'], final_weights['w_mobility'], 
                        final_weights['w_env'], final_weights['w_congestion']]
        colors = ['#00A36C', '#3498DB', '#9B59B6', '#E74C3C']
        
        wedges, texts, autotexts = ax1.pie(weight_values, labels=weight_names, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        for text in texts + autotexts:
            text.set_color('white')
            text.set_fontweight('bold')
        
        ax1.set_title('Final Optimized Weights', color='white', fontweight='bold', pad=20)
        
        # 2. Optimization convergence (if available)
        if opt_results and opt_results['results']:
            correlations = [r['correlation'] for r in opt_results['results']]
            ax2.plot(correlations, color='#4A90E2', linewidth=2, marker='o', markersize=3)
            ax2.fill_between(range(len(correlations)), correlations, alpha=0.3, color='#4A90E2')
            
            # Highlight best point
            best_idx = np.argmax(correlations)
            ax2.scatter([best_idx], [max(correlations)], color='#FFD700', s=100, zorder=5, 
                       label=f'Best: {max(correlations):.4f}')
            
            ax2.set_title(f'Optimization Convergence ({opt_results["method"]})', 
                         color='white', fontweight='bold')
            ax2.set_xlabel('Iteration', color='white')
            ax2.set_ylabel('Correlation with Target', color='white')
            ax2.tick_params(colors='white')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.spines['bottom'].set_color('white')
            ax2.spines['left'].set_color('white')
            ax2.spines['top'].set_color('none')
            ax2.spines['right'].set_color('none')
        
        self.weights_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Enhanced QoS Routing Optimizer")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Network Optimization Labs")
    
    window = EnhancedQoSApp()
    window.show()
    
    sys.exit(app.exec())