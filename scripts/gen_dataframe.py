"""
Generate a Dataframe with clinical notes
"""
import pandas as pd

def gen_dataframe():
        
    # Define the clinical notes
    clinical_notes = [
        "The patient is a 55-year-old male diagnosed with hypertension. He presents with persistent headaches, episodes of dizziness, and blurred vision. His current medication regimen includes Lisinopril, Amlodipine, and Hydrochlorothiazide.",
        "A 62-year-old female with a history of type 2 diabetes mellitus reports polyuria, polydipsia, and ongoing fatigue. She is being treated with Metformin, Glipizide, and Insulin.",
        "The patient is a 25-year-old male with asthma, experiencing wheezing, shortness of breath, and chest tightness. He is prescribed Albuterol, Fluticasone, and Montelukast.",
        "A 48-year-old female presents with rheumatoid arthritis, reporting joint pain, morning stiffness, and swelling. Her treatment includes Methotrexate, Hydroxychloroquine, and Prednisone.",
        "The patient is a 70-year-old male with chronic obstructive pulmonary disease (COPD), suffering from chronic cough, dyspnea, and frequent respiratory infections. His medications include Tiotropium, Salbutamol, and Prednisolone.",
        "A 35-year-old female with systemic lupus erythematosus presents with a facial rash, joint pain, and fatigue. She is currently taking Hydroxychloroquine, Prednisone, and Mycophenolate mofetil.",
        "The patient is a 45-year-old male with gastroesophageal reflux disease (GERD), experiencing heartburn, regurgitation, and chest pain. He is being treated with Omeprazole, Ranitidine, and Metoclopramide.",
        "A 29-year-old female diagnosed with irritable bowel syndrome (IBS) reports abdominal pain, bloating, and alternating diarrhea and constipation. Her medications include Dicyclomine, Loperamide, and Amitriptyline.",
        "The patient is a 60-year-old male with Parkinson’s disease, presenting with tremors, rigidity, and bradykinesia. He is prescribed Levodopa, Carbidopa, and Pramipexole.",
        "A 50-year-old female with chronic kidney disease (CKD) reports fatigue, swelling in her legs, and decreased urine output. Her treatment includes Lisinopril, Furosemide, and Erythropoietin.",
        "The patient is a 33-year-old male with ulcerative colitis, experiencing abdominal pain, bloody diarrhea, and weight loss. He is being treated with Mesalamine, Prednisone, and Azathioprine.",
        "A 47-year-old female with fibromyalgia presents with widespread musculoskeletal pain, fatigue, and sleep disturbances. Her medications include Pregabalin, Duloxetine, and Tramadol.",
        "The patient is a 55-year-old male diagnosed with diabetes, suffering from obesity, fatigue, and swelling in the legs. His current medication regimen includes Insulin, Sulfonylureas (SU), and Carvedilol.",
        "A 30-year-old female with multiple sclerosis (MS) reports numbness, muscle weakness, and visual disturbances. She is currently taking Interferon beta, Glatiramer acetate, and Methylprednisolone.",
        "The patient is a 70-year-old female with osteoporosis, experiencing back pain, a decrease in height, and a recent hip fracture. Her treatment includes Alendronate, Calcium, and Vitamin D supplements.",
        "A 52-year-old male diagnosed with hepatitis C presents with fatigue, jaundice, and abdominal pain. He is being treated with Sofosbuvir, Ribavirin, and Peginterferon alfa.",
        "The patient is a 40-year-old female with migraine, reporting severe headache, nausea, and sensitivity to light. She is prescribed Sumatriptan, Propranolol, and Topiramate.",
        "A 65-year-old male with benign prostatic hyperplasia (BPH) experiences difficulty urinating, nocturia, and a weak urine stream. His medications include Tamsulosin, Finasteride, and Dutasteride.",
        "The patient is a 28-year-old female with polycystic ovary syndrome (PCOS), suffering from irregular periods, hirsutism, and acne. She is currently taking Metformin, Spironolactone, and oral contraceptives.",
        "A 58-year-old male diagnosed with chronic hepatitis B presents with fatigue, abdominal pain, and jaundice. He is being treated with Tenofovir, Entecavir, and Lamivudine.",
        "The patient is a 45-year-old female with hypothyroidism, experiencing weight gain, fatigue, and cold intolerance. Her treatment includes Levothyroxine, Liothyronine, and Selenium supplements.",
        "A 38-year-old male with celiac disease reports diarrhea, weight loss, and bloating. His medications include a gluten-free diet, Loperamide, and Vitamin D supplements.",
        "The patient is a 50-year-old female with psoriasis, presenting with red, scaly patches on her skin, itching, and joint pain. She is currently taking Methotrexate, Adalimumab, and Calcipotriol.",
        "A 60-year-old male diagnosed with gout experiences severe joint pain, swelling, and redness, primarily in his big toe. He is being treated with Allopurinol, Colchicine, and Indomethacin.",
        "The patient is a 27-year-old female with endometriosis, suffering from pelvic pain, dysmenorrhea, and infertility. Her medications include Danazol, Leuprolide, and oral contraceptives.",
        "A 55-year-old male with chronic hepatitis C presents with fatigue, abdominal pain, and jaundice. He is being treated with Sofosbuvir, Ledipasvir, and Ribavirin.",
        "The patient is a 40-year-old female with Crohn’s disease, experiencing abdominal pain, diarrhea, and weight loss. Her treatment includes Infliximab, Azathioprine, and Mesalamine.",
        "A 68-year-old male diagnosed with prostate cancer reports difficulty urinating, hematuria, and pelvic pain. He is currently taking Leuprolide, Flutamide, and Bicalutamide.",
        "The patient is a 33-year-old female with ankylosing spondylitis, suffering from chronic back pain, stiffness, and reduced mobility. Her medications include Etanercept, Sulfasalazine, and Naproxen.",
        "A 50-year-old male with amyotrophic lateral sclerosis (ALS) presents with muscle weakness, difficulty speaking, and respiratory problems. His treatment includes Riluzole, Baclofen, and Edaravone."
    ]

    # Create a list of Patient IDs and codes
    patient_ids = [f"PID{1000+i}" for i in range(len(clinical_notes))]
    codes = [f"CODE{1000+i}" for i in range(len(clinical_notes))]

    # Create a DataFrame
    data = {
        "patient_id": patient_ids,
        "code_id": codes,
        "clinical_notes": clinical_notes
    }

    df = pd.DataFrame(data)

    df.to_csv("dataset_notes/clin_note_df.tsv", sep="\t", index=False)


if __name__ == "__main__":
    gen_dataframe()

