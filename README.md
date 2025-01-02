
# **CpG Site Detection \- LSTM Implementation**
## **Project Description**
Deep learning solution for counting CpG sites in DNA sequences using LSTM architecture. Processes DNA sequences containing N, A, C, G, T nucleotides and predicts the count of CG dimers.
## **Requirements**
* Python 3.8+  
* PyTorch 2.0+  
* NumPy  
* Streamlit (for web interface)
## **Installation**
bash  
Copy  
git clone \[repository-url\]  
cd cpg-detector
pip install \-r requirements.txt
## **Usage**
### **Training**
python  
Copy
python model.py
### **Web Interface**
python  
Copy
streamlit run app.py
## **Model Architecture**
* 2-layer LSTM  
* Hidden size: 128  
* Dropout: 0.2  
* Dense layers: 128→64→1
## **Performance**
* Training samples: 2048  
* Test samples: 512  
* Average prediction error: ±1 CpG site
