# Academic Document Verification System
# Main application file: app.py

import os
import re
import numpy as np
import pandas as pd
import cv2
import pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import hashlib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import streamlit as st
import io
import base64
from pdf2image import convert_from_path, convert_from_bytes
import sqlite3
import json
import traceback

# Initialize database
def init_db():
    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS verified_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prn TEXT,
        student_name TEXT, 
        college_name TEXT,
        verification_date TEXT,
        verification_status TEXT,
        confidence_score REAL,
        document_hash TEXT,
        metadata TEXT
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS verification_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        document_id INTEGER,
        action TEXT,
        details TEXT,
        FOREIGN KEY (document_id) REFERENCES verified_documents (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Document processor class
class DocumentProcessor:
    def _init_(self):
        self.extraction_confidence = 0
        self.verification_score = 0
        self.anomaly_score = 0
        self.validation_results = {}
        
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale if the image has color channels
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA image
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                elif image.shape[2] == 3:  # RGB image
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image  # Unknown format, use as-is
            else:
                gray = image  # Already grayscale

            # Apply thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Remove noise
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return opening
        except Exception as e:
            st.error(f"Error during image preprocessing: {str(e)}")
            return image  # Return original image if preprocessing fails

    def pdf_to_image(self, pdf_bytes):
        """Convert PDF bytes to image"""
        try:
            images = convert_from_bytes(pdf_bytes)
            return np.array(images[0])
        except Exception as e:
            if "poppler" in str(e).lower():
                raise Exception("Poppler utilities not found. Please install Poppler for PDF processing.")
            else:
                raise e
    
    def process_pdf_alternative(pdf_bytes):
        import fitz  # PyMuPDF
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        first_page = pdf_document.load_page(0)
        pix = first_page.get_pixmap()
        image_data = pix.tobytes("png")
        return np.array(Image.open(io.BytesIO(image_data)))
    
    def extract_text(self, image):
        """Extract text from image using OCR"""
        try:
            preprocessed = self.preprocess_image(image)
            text = pytesseract.image_to_string(preprocessed)
            
            # Calculate confidence based on OCR
            confidence_data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
            confidences = [float(conf) for conf in confidence_data['conf'] if conf != '-1']
            self.extraction_confidence = np.mean(confidences) if confidences else 0
            
            return text
        except Exception as e:
            st.error(f"OCR extraction error: {str(e)}")
            st.error("Make sure Tesseract OCR is properly installed and in your PATH")
            return ""
    
    def extract_fields(self, text):
        """Extract relevant fields from OCR text"""
        fields = {
            'prn': None,
            'student_name': None,
            'mother_name': None,
            'college_name': None,
            'branch': None,
            'subjects': [],
            'grades': [],
            'credits': [],
            'sgpa': None,
            'result_date': None
        }
        
        # Debug: Show extracted text
        st.subheader("Extracted Text (Debug)")
        st.text(text)
        
        # Extract PRN
        prn_match = re.search(r'Perm\s+Reg\s+No\(PRN\):\s*([0-9A-Za-z]+)', text)
        if prn_match:
            fields['prn'] = prn_match.group(1)
        
        # Extract Student Name
        name_match = re.search(r'Student\s+Name:\s*([^\n]+)', text)
        if name_match:
            fields['student_name'] = name_match.group(1).strip()
        
        # Extract Mother's Name
        mother_match = re.search(r'Mother\s+Name:\s*([^\n]+)', text)
        if mother_match:
            fields['mother_name'] = mother_match.group(1).strip()
        
        # Extract College Name
        college_match = re.search(r'College\s+Name:\s*[0-9]+\s+([^\n]+)', text)
        if college_match:
            fields['college_name'] = college_match.group(1).strip()
        
        # Extract Branch/Course
        branch_match = re.search(r'Branch/Course:\s*([^\n]+)', text)
        if branch_match:
            fields['branch'] = branch_match.group(1).strip()
        
        # Extract SGPA
        sgpa_match = re.search(r'SGPA\s+[0-9]+\s*:-\s*([0-9.]+)', text)
        if sgpa_match:
            fields['sgpa'] = float(sgpa_match.group(1))
        
        # Extract Result Date
        date_match = re.search(r'RESULT\s+DATE:\s*([0-9]+\s+[A-Za-z]+\s+[0-9]+)', text)
        if date_match:
            fields['result_date'] = date_match.group(1).strip()
        
        # Extract subjects, grades and credits
        # Looking for patterns like: "310241 DATABASE MANAGEMENT SYSTEMS 3 A+ 27"
        subject_pattern = r'([0-9]{6})\s+([A-Z\s.&]+)\s+([0-9])\s+([A-Z+]+)\s+([0-9]+)'
        subject_matches = re.finditer(subject_pattern, text)
        
        for match in subject_matches:
            subject_code = match.group(1)
            subject_name = match.group(2).strip()
            credits = match.group(3)
            grade = match.group(4)
            grade_points = match.group(5)
            
            fields['subjects'].append({
                'code': subject_code,
                'name': subject_name,
                'credits': int(credits),
                'grade': grade,
                'grade_points': int(grade_points)
            })
        
        return fields
    
    def verify_document(self, fields):
        """Verify document authenticity using various checks"""
        verification_checks = {
            'prn_format': False,
            'grade_validation': False,
            'credit_sum': False,
            'sgpa_calculation': False,
            'date_format': False
        }
        
        # PRN format check
        if fields['prn'] and re.match(r'^[0-9A-Za-z]+$', fields['prn']):
            verification_checks['prn_format'] = True
        
        # Grade validation
        valid_grades = {'O', 'A+', 'A', 'B+', 'B', 'C', 'D', 'F', 'P'}
        if fields['subjects']:
            all_valid = all(subject['grade'] in valid_grades for subject in fields['subjects'])
            verification_checks['grade_validation'] = all_valid
        
        # Credit sum check
        if fields['subjects']:
            total_credits = sum(subject['credits'] for subject in fields['subjects'])
            verification_checks['credit_sum'] = (total_credits > 0)
        
        # SGPA calculation check
        if fields['subjects'] and fields['sgpa']:
            total_grade_points = sum(subject['grade_points'] for subject in fields['subjects'])
            total_credits = sum(subject['credits'] for subject in fields['subjects'])
            calculated_sgpa = round(total_grade_points / total_credits, 2)
            verification_checks['sgpa_calculation'] = (abs(calculated_sgpa - fields['sgpa']) < 0.1)
        
        # Date format check
        if fields['result_date']:
            date_pattern = r'[0-9]+\s+[A-Za-z]+\s+[0-9]+'
            verification_checks['date_format'] = bool(re.match(date_pattern, fields['result_date']))
        
        self.validation_results = verification_checks
        
        # Calculate verification score
        passed_checks = sum(verification_checks.values())
        total_checks = len(verification_checks)
        self.verification_score = passed_checks / total_checks if total_checks > 0 else 0
        
        return verification_checks
    
    def detect_anomalies(self, fields):
        """Detect anomalies in the document using ML techniques"""
        if not fields['subjects']:
            self.anomaly_score = 0.5  # Neutral if no subjects
            return 0.5
        
        # Prepare data for anomaly detection
        subject_data = []
        for subject in fields['subjects']:
            grade_value = {'O': 10, 'A+': 9, 'A': 8, 'B+': 7, 'B': 6, 'C': 5, 'D': 4, 'F': 0, 'P': 0}
            subject_data.append([
                subject['credits'],
                grade_value.get(subject['grade'], 0),
                subject['grade_points']
            ])
        
        if len(subject_data) < 3:  # Not enough data for meaningful detection
            self.anomaly_score = 0.5
            return 0.5
        
        # Use Isolation Forest for anomaly detection
        X = np.array(subject_data)
        clf = IsolationForest(random_state=42, contamination=0.1)
        clf.fit(X)
        
        # Get anomaly scores (-1 for anomalies, 1 for normal)
        anomaly_predictions = clf.predict(X)
        anomaly_score = 1 - (len(anomaly_predictions[anomaly_predictions == -1]) / len(anomaly_predictions))
        
        self.anomaly_score = anomaly_score
        return anomaly_score
    
    def calculate_final_confidence(self):
        """Calculate final confidence score based on all metrics"""
        # Weights for different components
        extraction_weight = 0.3
        verification_weight = 0.5
        anomaly_weight = 0.2
        
        confidence = (
            extraction_weight * (self.extraction_confidence / 100) +
            verification_weight * self.verification_score +
            anomaly_weight * self.anomaly_score
        )
        
        return min(max(confidence, 0), 1)  # Ensure between 0 and 1
    
    def generate_document_hash(self, image):
        """Generate a unique hash for the document"""
        return hashlib.sha256(image.tobytes()).hexdigest()

# Database operations
def save_verification_result(fields, confidence, document_hash, verification_status):
    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    
    # Check if document already exists
    c.execute("SELECT id FROM verified_documents WHERE document_hash = ?", (document_hash,))
    result = c.fetchone()
    
    if result:
        # Document exists, update it
        doc_id = result[0]
        c.execute("""
        UPDATE verified_documents SET 
            verification_date = ?, 
            verification_status = ?, 
            confidence_score = ?
        WHERE id = ?
        """, (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            verification_status,
            confidence,
            doc_id
        ))
    else:
        # Insert new document
        metadata = json.dumps({
            'subjects': fields.get('subjects', []),
            'sgpa': fields.get('sgpa'),
            'branch': fields.get('branch'),
            'result_date': fields.get('result_date')
        })
        
        c.execute("""
        INSERT INTO verified_documents 
        (prn, student_name, college_name, verification_date, verification_status, confidence_score, document_hash, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fields.get('prn', ''),
            fields.get('student_name', ''),
            fields.get('college_name', ''),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            verification_status,
            confidence,
            document_hash,
            metadata
        ))
        
        doc_id = c.lastrowid
    
    # Log the verification action
    c.execute("""
    INSERT INTO verification_logs
    (timestamp, document_id, action, details)
    VALUES (?, ?, ?, ?)
    """, (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        doc_id,
        "VERIFICATION",
        f"Document verified with confidence: {confidence:.2f}"
    ))
    
    conn.commit()
    conn.close()
    return doc_id

def get_verification_history(limit=10):
    conn = sqlite3.connect('document_verification.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("""
    SELECT id, prn, student_name, college_name, verification_date, verification_status, confidence_score
    FROM verified_documents
    ORDER BY verification_date DESC
    LIMIT ?
    """, (limit,))
    
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    return results

def get_document_by_prn(prn):
    conn = sqlite3.connect('document_verification.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("""
    SELECT * FROM verified_documents
    WHERE prn = ?
    ORDER BY verification_date DESC
    LIMIT 1
    """, (prn,))
    
    result = c.fetchone()
    if result:
        document = dict(result)
        document['metadata'] = json.loads(document['metadata'])
        return document
    
    conn.close()
    return None

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Academic Document Verification System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize database
    init_db()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Verify Document", "History", "Analytics", "About"])
    
    if page == "Verify Document":
        show_verification_page()
    elif page == "History":
        show_history_page()
    elif page == "Analytics":
        show_analytics_page()
    else:
        show_about_page()

def show_verification_page():
    st.title("Academic Document Verification System")
    st.write("Upload an academic document (marksheet) for verification.")
    
    # Add system requirements information
    with st.expander("System Requirements"):
        st.markdown("""
        For this system to work properly, please ensure you have:
        1. *Tesseract OCR*: Required for text extraction
            - Windows: [Download Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
            - Linux: sudo apt-get install tesseract-ocr
            - macOS: brew install tesseract
        
        2. *Poppler* (for PDF processing):
            - Windows: [Download Poppler](https://github.com/oschwartz10612/poppler-windows/releases/)
            - Linux: sudo apt-get install poppler-utils
            - macOS: brew install poppler
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Show processing indicator
        with st.spinner("Processing document..."):
            # Process the document
            try:
                # Convert to image
                file_bytes = uploaded_file.read()
                processor = DocumentProcessor()
                
                if uploaded_file.type == "application/pdf":
                    try:
                        image = processor.pdf_to_image(file_bytes)
                    except Exception as e:
                        if "poppler" in str(e).lower():
                            st.error("Poppler utilities not found. Please install Poppler to process PDF files.")
                            st.markdown("""
                            ### How to install Poppler:
                            
                            *Windows:*
                            1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/
                            2. Extract the ZIP file
                            3. Add the bin directory to your system PATH
                            
                            *macOS:*
                            
                            brew install poppler
                            
                            
                            *Linux:*
                            
                            sudo apt-get install poppler-utils
                            
                            
                            Alternatively, you can upload a PNG or JPG image of your document.
                            """)
                            return
                        else:
                            st.error(f"PDF processing error: {str(e)}")
                            return
                else:
                    try:
                        # Use PILlow to open the image
                        image = np.array(Image.open(io.BytesIO(file_bytes)))
                        
                        # Debug: Show image shape and type
                        st.write(f"Image shape: {image.shape}, dtype: {image.dtype}")
                    except Exception as e:
                        st.error(f"Error opening image: {str(e)}")
                        return
                
                # Display the uploaded document - Fix deprecated parameter
                st.image(image, caption="Uploaded Document", use_container_width=True)
                
                # Extract text from image
                text = processor.extract_text(image)
                
                if not text or len(text.strip()) < 10:  # Check if OCR failed
                    st.error("Text extraction failed. The image may be unclear or Tesseract OCR may not be properly installed.")
                    st.write("Make sure Tesseract OCR is installed and properly configured.")
                    return
                
                # Extract fields from text
                fields = processor.extract_fields(text)
                
                # Verify document
                verification_results = processor.verify_document(fields)
                
                # Detect anomalies
                anomaly_score = processor.detect_anomalies(fields)
                
                # Calculate final confidence
                confidence = processor.calculate_final_confidence()
                
                # Generate document hash
                document_hash = processor.generate_document_hash(image)
                
                # Determine verification status
                if confidence >= 0.8:
                    verification_status = "VERIFIED"
                elif confidence >= 0.5:
                    verification_status = "NEEDS REVIEW"
                else:
                    verification_status = "POTENTIALLY FRAUDULENT"
                
                # Save to database
                save_verification_result(fields, confidence, document_hash, verification_status)
                
                # Display results
                st.header("Verification Results")
                
                # Create two columns
                col1, col2 = st.columns(2)
                
                # Display basic document info
                with col1:
                    st.subheader("Document Information")
                    st.write(f"*Student Name:* {fields['student_name'] or 'Not detected'}")
                    st.write(f"*PRN:* {fields['prn'] or 'Not detected'}")
                    st.write(f"*College:* {fields['college_name'] or 'Not detected'}")
                    st.write(f"*Branch/Course:* {fields['branch'] or 'Not detected'}")
                    st.write(f"*SGPA:* {fields['sgpa'] or 'Not detected'}")
                    st.write(f"*Result Date:* {fields['result_date'] or 'Not detected'}")
                
                # Display verification results
                with col2:
                    st.subheader("Verification Status")
                    
                    # Show verification status with color
                    if verification_status == "VERIFIED":
                        st.markdown(f"<h3 style='color:green;'>{verification_status}</h3>", unsafe_allow_html=True)
                    elif verification_status == "NEEDS REVIEW":
                        st.markdown(f"<h3 style='color:orange;'>{verification_status}</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color:red;'>{verification_status}</h3>", unsafe_allow_html=True)
                    
                    # Display confidence score with gauge
                    st.write(f"*Confidence Score:* {confidence:.2f}")
                    
                    # Create a simple gauge visualization
                    gauge_color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.5 else "red"
                    st.progress(confidence)
                    
                    # Display document hash
                    st.write(f"*Document Hash:* {document_hash[:10]}...{document_hash[-10:]}")
                
                # Display subjects table
                if fields['subjects']:
                    st.subheader("Subject Information")
                    
                    # Create dataframe for subjects
                    subjects_df = pd.DataFrame([
                        {
                            "Subject Code": s["code"],
                            "Subject Name": s["name"],
                            "Credits": s["credits"],
                            "Grade": s["grade"],
                            "Grade Points": s["grade_points"]
                        }
                        for s in fields['subjects']
                    ])
                    
                    st.dataframe(subjects_df)
                else:
                    st.warning("No subject information detected in the document.")
                
                # Display verification checks
                st.subheader("Verification Checks")
                
                # Create check results table
                checks_data = []
                for check, passed in verification_results.items():
                    check_name = check.replace('_', ' ').title()
                    status = "✅ Passed" if passed else "❌ Failed"
                    checks_data.append({"Check": check_name, "Status": status})
                
                checks_df = pd.DataFrame(checks_data)
                st.dataframe(checks_df)
                
                # Anomaly detection results
                st.subheader("Anomaly Detection")
                st.write(f"*Anomaly Score:* {anomaly_score:.2f}")
                st.write("Lower score indicates higher probability of anomalies")
                
                # Display OCR confidence
                st.write(f"*OCR Extraction Confidence:* {processor.extraction_confidence:.2f}%")
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.error(traceback.format_exc())  # Show detailed error for debugging
                st.write("Please upload a clear document image for better processing.")

def show_history_page():
    st.title("Verification History")
    
    # Get verification history
    history = get_verification_history(50)
    
    if not history:
        st.info("No verification history found.")
    else:
        # Search by PRN
        search_prn = st.text_input("Search by PRN")
        if search_prn:
            filtered_history = [h for h in history if search_prn.lower() in h['prn'].lower()]
        else:
            filtered_history = history
        
        # Display history table
        if filtered_history:
            history_df = pd.DataFrame([
                {
                    "ID": h["id"],
                    "PRN": h["prn"],
                    "Student Name": h["student_name"],
                    "Verification Date": h["verification_date"],
                    "Status": h["verification_status"],
                    "Confidence": f"{h['confidence_score']:.2f}"
                }
                for h in filtered_history
            ])
            
            st.dataframe(history_df)
            
            # Select document for details
            if len(filtered_history) > 0:
                selected_id = st.selectbox(
                    "Select a document to view details", 
                    options=[h["id"] for h in filtered_history],
                    format_func=lambda x: f"ID: {x} - {next((h['student_name'] for h in filtered_history if h['id'] == x), '')}"
                )
                
                selected_doc = next((h for h in filtered_history if h["id"] == selected_id), None)
                
                if selected_doc:
                    st.subheader(f"Document Details: {selected_doc['student_name']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"*PRN:* {selected_doc['prn']}")
                        st.write(f"*College:* {selected_doc['college_name']}")
                    
                    with col2:
                        st.write(f"*Verification Date:* {selected_doc['verification_date']}")
                        
                        # Display status with color
                        status = selected_doc['verification_status']
                        if status == "VERIFIED":
                            st.markdown(f"*Status:* <span style='color:green;'>{status}</span>", unsafe_allow_html=True)
                        elif status == "NEEDS REVIEW":
                            st.markdown(f"*Status:* <span style='color:orange;'>{status}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"*Status:* <span style='color:red;'>{status}</span>", unsafe_allow_html=True)
                        
                        st.write(f"*Confidence:* {selected_doc['confidence_score']:.2f}")
        else:
            st.info("No records matching your search criteria.")

def show_analytics_page():
    st.title("Verification Analytics")
    
    # Get verification history
    history = get_verification_history(100)
    
    if not history:
        st.info("No verification data available for analytics.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Verification status distribution
        st.subheader("Verification Status Distribution")
        status_counts = df['verification_status'].value_counts()
        
        # Create and display pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {'VERIFIED': 'green', 'NEEDS REVIEW': 'orange', 'POTENTIALLY FRAUDULENT': 'red'}
        status_colors = [colors.get(status, 'gray') for status in status_counts.index]
        
        ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=status_colors)
        ax.axis('equal')
        st.pyplot(fig)
    
    with col2:
        # Confidence score distribution
        st.subheader("Confidence Score Distribution")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df['confidence_score'], bins=10, kde=True, ax=ax)
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    # Time series of verifications
    st.subheader("Verification Activity Over Time")
    
    # Convert verification_date to datetime
    df['verification_date'] = pd.to_datetime(df['verification_date'])
    df['date'] = df['verification_date'].dt.date
    
    # Group by date and count
    daily_counts = df.groupby('date').size().reset_index(name='count')
    
    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=daily_counts, x='date', y='count', marker='o', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Verifications')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    # College distribution
    st.subheader("College Distribution")
    
    college_counts = df['college_name'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=college_counts.values, y=college_counts.index, ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('College')
    st.pyplot(fig)

def show_about_page():
    st.title("About the Document Verification System")
    
    st.write("""
    ## Project Overview
    
    This Academic Document Verification System uses data science and machine learning techniques to verify the authenticity of academic documents, specifically focusing on marksheets and transcripts.
    
    ### Key Features:
    
    1. *Document Analysis*: Uses OCR to extract text and information from uploaded documents.
    
    2. *Data Validation*: Performs multiple validation checks to ensure document correctness.
    
    3. *Anomaly Detection*: Uses machine learning algorithms to identify unusual patterns or potential fraud.
    
    4. *Verification History*: Maintains a searchable database of previously verified documents.
    
    5. *Analytics Dashboard*: Provides insights into verification trends and statistics.
    
    ### Technical Implementation:
    
    - *OCR*: Uses Tesseract for text extraction from images
    - *Machine Learning*: Implements Isolation Forest for anomaly detection
    - *Database*: SQLite for document and verification history storage
    - *UI*: Streamlit for an intuitive user interface
    
    ### How it Works:
    
    1. Upload a document image or PDF
    2. System extracts information using OCR
    3. Multiple validation checks are performed
    4. Machine learning models detect anomalies
    5. Final verification status and confidence score are provided
    
    ### Best Practices for Document Uploads:
    
    - Use clear, high-resolution scans
    - Ensure the document is properly aligned
    - Make sure all text is clearly visible
    - PDF format is preferred for multi-page documents
    """)
    
    st.subheader("Data Science Techniques Used")
    
    techniques = [
        ("Optical Character Recognition (OCR)", "Extracts text data from document images"),
        ("Natural Language Processing", "Identifies and extracts key fields from unstructured text"),
        ("Anomaly Detection", "Uses Isolation Forest algorithm to identify unusual patterns"),
        ("Statistical Validation", "Performs statistical checks on extracted data"),
        ("Data Visualization", "Presents analytics and insights through interactive charts")
    ]
    
    techniques_df = pd.DataFrame(techniques, columns=["Technique", "Description"])
    st.table(techniques_df)

if _name_ == "_main_":
    main()
