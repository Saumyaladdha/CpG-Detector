import streamlit as st
import torch
from model import CpGPredictor, count_cpgs, dnaseq_to_intseq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CpGPredictor().to(device)
model.load_state_dict(torch.load('cpg_model.pth'))
model.eval()

def main():
    st.title("CpG Site Counter")
    sequence = st.text_input("Enter DNA sequence:", "NCACANNTNCGGAGGCGNA")

    if st.button("Count CpGs"):
        int_seq = list(dnaseq_to_intseq(sequence))
        input_tensor = torch.tensor([int_seq]).to(device)
        
        with torch.no_grad():
            prediction = model(input_tensor)
        
        st.write(f"Predicted CpG count: {prediction.item():.2f}")
        st.write(f"Actual CpG count: {count_cpgs(sequence)}")

if __name__ == "__main__":
    main()