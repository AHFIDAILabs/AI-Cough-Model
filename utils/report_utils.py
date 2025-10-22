import os
import base64
from fpdf import FPDF

os.makedirs("static/reports", exist_ok=True)

def clean_text_for_pdf(text):
    text = text.replace('\u2014', '-').replace('\u2013', '-').replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.encode('ascii', errors='ignore').decode('ascii')
    return text

def generate_pdf_report(result, filename):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="TB Risk Prediction Report", ln=True, align="C")
        pdf.ln(10)

        for key, value in result.items():
            if key not in ['spectrogram_base64', 'report_link', 'recommendation', 'disclaimer', 'debug_info']:
                clean_value = clean_text_for_pdf(str(value))
                pdf.cell(200, 10, txt=f"{key.capitalize()}: {clean_value}", ln=True)

        # Insert spectrogram if available
        if result.get('spectrogram_base64'):
            try:
                img_path = f"static/reports/{filename}_spectrogram.png"
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(result['spectrogram_base64']))
                pdf.image(img_path, w=160)
                os.remove(img_path)
            except Exception as e:
                print(f"Error adding spectrogram to PDF: {e}")

        pdf.ln(5)
        # Disclaimer
        if result.get('disclaimer'):
            pdf.set_font("Arial", size=11, style='I')
            pdf.multi_cell(190, 10, txt=f"Disclaimer: {clean_text_for_pdf(result['disclaimer'])}")

        # Recommendation section
        if result.get('recommendation'):
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Medical Recommendation", ln=True)
            pdf.ln(5)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(190, 6, txt=clean_text_for_pdf(result['recommendation']))

        report_path = f"static/reports/{filename}.pdf"
        pdf.output(report_path, "F")

        download_link = f"/static/reports/{filename}.pdf"
        result["report_link"] = download_link

        return report_path

    except Exception as e:
        print(f"PDF generation error: {e}")
        return None
