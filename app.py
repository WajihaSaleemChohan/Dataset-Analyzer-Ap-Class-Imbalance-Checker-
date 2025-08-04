import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import zipfile
import tarfile
from pathlib import Path
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
from kaggle.api.kaggle_api_extended import KaggleApi

import os
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import zipfile
import tarfile
from pathlib import Path
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import io

# Constants
TEMP_FOLDER = 'temp_folder'
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
VALID_ARCHIVE_EXTENSIONS = ('.zip', '.tar.gz', '.tgz')
KAGGLE_API_URL = "https://www.kaggle.com/api/v1"

# Kaggle API configuration using your secrets
KAGGLE_CREDS = {
    "username": st.secrets["kaggle"]["username"],
    "token": st.secrets["kaggle"]["auth_token"]
}

headers = {
    "Authorization": f"Bearer {KAGGLE_CREDS['token']}",
    "Content-Type": "application/json"
}

def download_kaggle_dataset(dataset_name):
    """Download dataset from Kaggle using direct API calls with authentication"""
    try:
        create_temp_folder()
        
        # 1. First get the dataset metadata to verify access
        dataset_info_url = f"{KAGGLE_API_URL}/datasets/view/{dataset_name}"
        info_response = requests.get(dataset_info_url, headers=headers)
        
        if info_response.status_code != 200:
            st.error(f"Failed to access dataset: {info_response.text}")
            return None
        
        # 2. Download the dataset
        st.info(f"Downloading dataset: {dataset_name}")
        download_url = f"{KAGGLE_API_URL}/datasets/download/{dataset_name}"
        
        with requests.get(download_url, headers=headers, stream=True) as response:
            response.raise_for_status()
            
            # Save the zip file
            zip_filename = f"{dataset_name.replace('/', '-')}.zip"
            zip_path = os.path.join(TEMP_FOLDER, zip_filename)
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(TEMP_FOLDER)
            
            # Remove the zip file
            os.remove(zip_path)
            
            st.success("Dataset downloaded and extracted successfully!")
            return TEMP_FOLDER
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return None

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #3498db;
    }
    .st-bb {
        background-color: #ffffff;
    }
    .st-at {
        background-color: #3498db;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .css-1v3fvcr {
        border-radius: 10px;
    }
    .st-eb {
        border-radius: 10px;
    }
    .st-cb {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================
def cleanup_temp_folder():
    """Remove temporary folder and its contents."""
    try:
        if os.path.exists(TEMP_FOLDER):
            for root, dirs, files in os.walk(TEMP_FOLDER, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(TEMP_FOLDER)
    except Exception as e:
        st.warning(f"Could not clean up temporary files: {e}")

def create_temp_folder():
    """Create temporary folder if it doesn't exist."""
    os.makedirs(TEMP_FOLDER, exist_ok=True)

# ==================== DATA PROCESSING FUNCTIONS ====================
def process_image_dataset(folder_path):
    """Process image dataset and return class distribution."""
    image_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)
    
    for root, dirs, files in os.walk(folder_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            if file.lower().endswith(VALID_IMAGE_EXTENSIONS):
                folder_name = os.path.basename(root)
                subclass = file.split('_')[0] if '_' in file else 'single_class'
                image_counts[folder_name][subclass] += 1
                total_counts[folder_name] += 1
    
    return image_counts, total_counts

def process_csv_dataset(csv_file_path):
    """Process CSV file and return class distribution."""
    try:
        df = pd.read_csv(csv_file_path)
        # Try to find a likely target column
        target_col = None
        for col in ['diagnosis', 'label', 'class', 'target']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            class_counts = df[target_col].value_counts()
            return class_counts, df.shape[0], target_col
        else:
            st.warning("No recognized target column found (tried: diagnosis, label, class, target)")
            return None, df.shape[0], None
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None, 0, None

# ==================== GENERAL DATA PROCESSING ====================
def process_general_dataset(file_path):
    """Process any dataset by extracting headers and first column entries"""
    try:
        df = pd.read_csv(file_path)
        
        # Extract main heading from first cell if available
        main_heading = "Dataset Analysis"
        if df.columns[0] and str(df.iloc[0,0]) != 'nan':
            main_heading = str(df.iloc[0,0])
            df = df.iloc[1:]  # Remove heading row if it exists
        
        # Get first column entries (excluding header)
        first_col_name = df.columns[0]
        entries = df[first_col_name].dropna().unique()
        
        # Get column headers (excluding first column)
        headers = [col for col in df.columns[1:] if str(col) != 'nan']
        
        st.subheader(main_heading)
        
        # Display basic information
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Number of {first_col_name}", len(entries))
        with col2:
            st.metric("Number of Columns", len(headers))
        
        # Show entries from first column
        with st.expander(f"View {first_col_name} List"):
            st.dataframe(pd.DataFrame({first_col_name: entries}))
        
        # Show column headers
        with st.expander("View Column Headers"):
            st.dataframe(pd.DataFrame({"Headers": headers}))
        
        # Show sample data
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        return True
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False

# ==================== FILE HANDLING FUNCTIONS ====================
def extract_archive(file_path, output_dir):
    """Extract zip or tar.gz file."""
    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif file_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(output_dir)
        return True
    except Exception as e:
        st.error(f"Failed to extract archive: {e}")
        return False

def download_kaggle_dataset(dataset_name):
    """Download dataset from Kaggle."""
    try:
        if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
            st.error("Kaggle API credentials not found. Please configure your Kaggle API.")
            return None
            
        kaggle.api.dataset_download_files(dataset_name, path=TEMP_FOLDER, unzip=True)
        return TEMP_FOLDER
    except Exception as e:
        st.error(f"Failed to download dataset: {e}")
        return None

# ==================== VISUALIZATION FUNCTIONS ====================
def display_hierarchy(image_counts, total_counts):
    """Display dataset hierarchy and class distribution."""
    if not total_counts:
        st.warning("No images found in the dataset")
        return
    
    st.subheader("Dataset Hierarchy")
    
    # Create a tree-like structure
    tree = {}
    for folder, counts in image_counts.items():
        tree[folder] = {
            'subclasses': dict(counts),
            'total': total_counts[folder]
        }
    
    # Display as expandable tree
    for folder, data in tree.items():
        with st.expander(f"📁 {folder} (Total: {data['total']})"):
            for subclass, count in data['subclasses'].items():
                st.write(f"├── {subclass}: {count}")
    
    # Show class distribution chart
    st.subheader("Class Distribution")
    df = pd.DataFrame.from_dict(total_counts, orient='index', columns=['Count'])
    df = df.sort_values('Count', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind='bar', ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def create_sunburst_chart(image_counts, total_counts):
    """Create interactive sunburst chart of folder structure."""
    paths = []
    names = []
    values = []
    
    # Add root level
    total_files = sum(total_counts.values())
    paths.append("")
    names.append("Dataset")
    values.append(total_files)
    
    # Add folders and subclasses
    for folder, counts in image_counts.items():
        paths.append("Dataset")
        names.append(folder)
        values.append(total_counts[folder])
        
        for subclass, count in counts.items():
            paths.append(folder)
            names.append(subclass)
            values.append(count)
    
    fig = px.sunburst(
        names=names,
        parents=paths,
        values=values,
        color=names,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        height=600
    )
    fig.update_traces(textinfo="label+percent entry")
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fig

def display_dashboard(image_counts, total_counts):
    """Display metrics dashboard."""
    total_files = sum(total_counts.values())
    num_folders = len(total_counts)
    ext_summary = defaultdict(int)
    
    for counts in image_counts.values():
        for ext, count in counts.items():
            ext_summary[ext] += count
    
    num_ext_types = len(ext_summary)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📂 Total Files", total_files)
    with col2:
        st.metric("📁 Folders", num_folders)
    with col3:
        st.metric("🖼️ Image Types", num_ext_types)
    
    style_metric_cards(background_color="#FFFFFF", border_left_color="#3498db")

def display_image_samples(folder_path, num_samples=3):
    """Display sample images from dataset."""
    image_files = [f for f in Path(folder_path).rglob('*') if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    if image_files:
        st.subheader("🖼️ Sample Images")
        cols = st.columns(num_samples)
        for idx, img_path in enumerate(image_files[:num_samples]):
            with cols[idx]:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)

# ==================== TAB FUNCTIONS ====================
def display_welcome():
    """Display attractive welcome section with app features"""
    st.title("📊 Data Explorer Pro")
    st.markdown("""
    <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;">
        <h3 style="color:#2c3e50;">🔍 Analyze, Visualize & Understand Your Datasets</h3>
        <p>Upload your datasets in multiple formats and get instant insights about:</p>
        <ul>
            <li>Class distribution</li>
            <li>Dataset hierarchy</li>
            <li>Data imbalances</li>
            <li>File statistics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📤 Archive Upload**\n\nZIP/TAR.GZ files")
    with col2:
        st.success("**🔍 Kaggle Datasets**\n\nDirect downloads")
    with col3:
        st.warning("**📊 CSV Analysis**\n\nClass distributions")
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        1. **For Archive Files**: Go to Upload Archive tab
        2. **For Kaggle Datasets**: Go to Kaggle Download tab
        3. **For CSV Files**: Go to CSV Analysis tab
        """)

def archive_upload_tab():
    """Enhanced archive upload with more features"""
    st.header("📤 Archive Upload", divider='rainbow')
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Drag & drop your dataset archive here",
        type=VALID_ARCHIVE_EXTENSIONS,
        help="Supported formats: ZIP, TAR.GZ, TGZ",
        key="archive_uploader"
    )
    
    if uploaded_file:
        # Add analysis options
        analysis_options = st.multiselect(
            "Select analysis types:",
            ["Basic Statistics", "Class Distribution", "Image Samples", "Data Quality"],
            default=["Basic Statistics", "Class Distribution"]
        )
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            create_temp_folder()
            file_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
            
            try:
                # Save uploaded file
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract archive
                if extract_archive(file_path, TEMP_FOLDER):
                    st.success("File uploaded and extracted successfully!")
                    
                    # Check if it's an image dataset
                    image_files = [f for f in Path(TEMP_FOLDER).rglob('*') if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
                    if image_files:
                        image_counts, total_counts = process_image_dataset(TEMP_FOLDER)
                        
                        if "Basic Statistics" in analysis_options:
                            display_dashboard(image_counts, total_counts)
                        
                        if "Class Distribution" in analysis_options:
                            col1, col2 = st.columns(2)
                            with col1:
                                display_hierarchy(image_counts, total_counts)
                            with col2:
                                st.plotly_chart(create_sunburst_chart(image_counts, total_counts), 
                                              use_container_width=True)
                        
                        if "Image Samples" in analysis_options:
                            display_image_samples(TEMP_FOLDER)
                    else:
                        st.warning("No image files found in the archive")
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")

def kaggle_download_tab():
    """Enhanced Kaggle download with dataset preview"""
    st.header("🔍 Kaggle Dataset Download", divider='rainbow')
    
    # Dataset search and selection
    dataset_name = st.text_input(
        "Enter Kaggle Dataset Name", 
        placeholder="username/dataset-name",
        help="Example: keras/cats-and-dogs"
    )
    
    if dataset_name:
        # Download and analyze
        if st.button("↓ Download & Analyze", type="primary"):
            with st.spinner(f"Downloading {dataset_name}..."):
                create_temp_folder()
                output_dir = download_kaggle_dataset(dataset_name)
                
                if output_dir:
                    # Check for images
                    image_files = [f for f in Path(output_dir).rglob('*') if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
                    if image_files:
                        image_counts, total_counts = process_image_dataset(output_dir)
                        
                        # Display results
                        st.success("✅ Dataset analysis complete!")
                        display_dashboard(image_counts, total_counts)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            display_hierarchy(image_counts, total_counts)
                        with col2:
                            st.plotly_chart(create_sunburst_chart(image_counts, total_counts), 
                                          use_container_width=True)
                        
                        display_image_samples(output_dir)
                    else:
                        st.warning("No image files found in the dataset")

def csv_analysis_tab():
    """Enhanced CSV analysis with automatic structure detection"""
    st.header("📊 CSV Data Analysis", divider='rainbow')
    
    uploaded_file = st.file_uploader(
        "Upload your data file", 
        type=['csv'],
        help="Supports any CSV file with headers"
    )
    
    if uploaded_file:
        create_temp_folder()
        file_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
        
        with st.spinner("Analyzing data structure..."):
            try:
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # First try the general processor
                if not process_general_dataset(file_path):
                    # Fall back to specialized processors if needed
                    st.info("Attempting specialized analysis...")
                    class_counts, total_entries, target_col = process_csv_dataset(file_path)
                    
                    if class_counts is not None:
                        st.subheader(f"Class Distribution (Detected Target: '{target_col}')")
                        st.dataframe(class_counts)
                        
                        fig = px.bar(
                            class_counts,
                            x=class_counts.index,
                            y=class_counts.values,
                            labels={'x': target_col, 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# ==================== MAIN APP ====================
def main():
    # Page config
    st.set_page_config(
        page_title="Data Explorer Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar with navigation
    with st.sidebar:
        st.image("https://cdn.prod.website-files.com/605c9e03d6553a5d82976ce2/65af45ee6d439082efd5d807_data-exploration-diagram.svg", 
             use_container_width=True)
        st.title("Navigation")
        selected_tab = st.radio(
            "Go to:",
            ["🏠 Home", "📤 Upload Archive", "🔍 Kaggle Dataset Download And Anlyzer", "📊 CSV Analysis"],
            label_visibility="collapsed"
        )
        
        # Settings
        st.title("Settings")
        auto_cleanup = st.checkbox("Auto cleanup temporary files", value=True)
    
    # Main content
    if selected_tab == "🏠 Home":
        display_welcome()
    elif selected_tab == "📤 Upload Archive":
        archive_upload_tab()
    elif selected_tab == "🔍 Kaggle Download":
        kaggle_download_tab()
    elif selected_tab == "📊 CSV Analysis":
        csv_analysis_tab()
    
    # Footer
    st.divider()
    st.caption("Data Explorer Pro v1.0 | © 2023")
    
    # Cleanup when done
    if auto_cleanup:
        cleanup_temp_folder()

if __name__ == '__main__':
    main()


