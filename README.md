📊 Dataset Analyzer App (Class Imbalance Checker)
🌟 Overview:
The Dataset Analyzer App is a powerful tool designed to help users analyze and visualize the distribution of classes in image datasets, especially for detecting class imbalances. Whether you're working with Kaggle datasets, ZIP archives, or CSV files, this app provides insights into the structure, class distribution, and image counts of your dataset. It offers interactive visualizations to explore the data easily.

🚀 Key Features:
⚖️ Class Imbalance Detection: Automatically checks for class imbalances in your dataset and displays class distribution across different folders or CSV target columns.
📊 Interactive Visualizations: Includes sunburst charts and bar charts to visualize the dataset's structure and class distribution.
🗂️ Flexible Dataset Support: Supports datasets in various formats:
📦 Kaggle: Download and analyze Kaggle datasets directly using the Kaggle API.
📂 ZIP Archives: Upload and extract image datasets stored in ZIP or TAR.GZ formats.
📑 CSV Files: Process CSV files for class distribution analysis and data exploration.
🖼️ Sample Image Display: Displays random sample images from the dataset for preview.

🔍 Dynamic CSV Analysis: Automatically detects target columns in CSV files, including numeric or string identifiers, and displays the class distribution.

🛠️ How It Works:
📥 Kaggle Dataset Download: Enter the Kaggle dataset name, and the app will download and unzip the dataset to a temporary folder for analysis.
📤 ZIP Archive Upload: Upload a ZIP/TAR.GZ archive containing images. The app will extract and process the images for class distribution.
📂 CSV File Upload: Upload a CSV file containing class labels. The app will identify the target column dynamically and generate a class distribution report.

💻 Technologies:
Streamlit: For creating the web interface.
Plotly: For interactive visualizations (sunburst and bar charts).
Pandas: For data handling and processing.
Kaggle API: For downloading datasets directly from Kaggle.
Matplotlib: For static plots.

🧩 Use Cases:
⚖️ Class Imbalance Detection: Identify whether certain classes dominate your dataset, which could impact model performance.
📊 Data Exploration: Quickly explore the structure of your dataset, including number of classes, images per class, and class distribution.
🏆 Kaggle Competitions: Easily check the class balance of your Kaggle dataset before starting the analysis or model training.

🛠️ Installation:
To run the app locally:

Clone the repository:

git clone https://github.com/yourusername/dataset-analyzer-app.git

Install the required dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

🧑‍💻 Contributing:
Feel free to fork the repository and submit pull requests for any improvements or bug fixes. Contributions are always welcome!

🌱 Future Improvements:
🚀 More visualization options (heatmaps, histograms, etc.)
🔄 Export reports in various formats (PDF, Excel)
🔧 Support for other dataset formats (e.g., JSON, Parquet)
