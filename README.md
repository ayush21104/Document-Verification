# Document-Verification
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
