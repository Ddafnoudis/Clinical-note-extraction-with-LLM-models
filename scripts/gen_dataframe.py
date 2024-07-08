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


    # Original DataFrame
    data2 = {
        'patient_id': ['PID1000', 'PID1001', 'PID1002', 'PID1003', 'PID1004', 
                    'PID1005', 'PID1006', 'PID1007', 'PID1008', 'PID1009', 
                    'PID1010', 'PID1011', 'PID1012', 'PID1013', 'PID1014', 
                    'PID1015', 'PID1016', 'PID1017', 'PID1018', 'PID1019', 
                    'PID1020', 'PID1021', 'PID1022', 'PID1023', 'PID1024', 
                    'PID1025', 'PID1026', 'PID1027', 'PID1028', 'PID1029'],
        'code_id': ['CODE1000', 'CODE1001', 'CODE1002', 'CODE1003', 'CODE1004', 
                    'CODE1005', 'CODE1006', 'CODE1007', 'CODE1008', 'CODE1009', 
                    'CODE1010', 'CODE1011', 'CODE1012', 'CODE1013', 'CODE1014', 
                    'CODE1015', 'CODE1016', 'CODE1017', 'CODE1018', 'CODE1019', 
                    'CODE1020', 'CODE1021', 'CODE1022', 'CODE1023', 'CODE1024', 
                    'CODE1025', 'CODE1026', 'CODE1027', 'CODE1028', 'CODE1029'],
        'clinical_notes': [
            'Patienten er en 55-årig mand diagnosticeret med hypertension. Han præsenterer med vedvarende hovedpine, episoder af svimmelhed og sløret syn. Hans nuværende medicinregime inkluderer Lisinopril, Amlodipin og Hydrochlorothiazid.',
            'En 62-årig kvinde med en historie med type 2 diabetes mellitus rapporterer polyuri, polydipsi og vedvarende træthed. Hun behandles med Metformin, Glipizid og Insulin.',
            'Patienten er en 25-årig mand med astma, der oplever hvæsen, åndenød og brystsmerter. Han er ordineret Albuterol, Fluticason og Montelukast.',
            'En 48-årig kvinde præsenterer med leddegigt, rapporterer ledsmerter, morgenstivhed og hævelse. Hendes behandling inkluderer Methotrexat, Hydroxychloroquin og Prednison.',
            'Patienten er en 70-årig mand med kronisk obstruktiv lungesygdom (KOL), lider af kronisk hoste, åndenød og hyppige luftvejsinfektioner. Hans medicin inkluderer Tiotropium, Salbutamol og Prednisolon.',
            'En 35-årig kvinde med systemisk lupus erythematosus præsenterer med et ansigtsudslæt, ledsmerter og træthed. Hun tager i øjeblikket Hydroxychloroquin, Prednison og Mycophenolat mofetil.',
            'Patienten er en 45-årig mand med gastroøsofageal reflukssygdom (GERD), oplever halsbrand, regurgitation og brystsmerter. Han behandles med Omeprazol, Ranitidin og Metoclopramid.',
            'En 29-årig kvinde diagnosticeret med irritabel tarmsyndrom (IBS) rapporterer mavesmerter, oppustethed og skiftevis diarré og forstoppelse. Hendes medicin inkluderer Dicyclomin, Loperamid og Amitriptylin.',
            'Patienten er en 60-årig mand med Parkinsons sygdom, præsenterer med rysten, stivhed og bradykinesi. Han er ordineret Levodopa, Carbidopa og Pramipexol.',
            'En 50-årig kvinde med kronisk nyresygdom (CKD) rapporterer træthed, hævelse i benene og nedsat urinproduktion. Hendes behandling inkluderer Lisinopril, Furosemid og Erythropoietin.',
            'Patienten er en 33-årig mand med colitis ulcerosa, oplever mavesmerter, blodig diarré og vægttab. Han behandles med Mesalamin, Prednison og Azathioprin.',
            'En 47-årig kvinde med fibromyalgi præsenterer med udbredt muskuloskeletal smerte, træthed og søvnforstyrrelser. Hendes medicin inkluderer Pregabalin, Duloxetin og Tramadol.',
            'Patienten er en 55-årig mand diagnosticeret med diabetes, lider af fedme, træthed og hævelse i benene. Hans nuværende medicinregime inkluderer Insulin, Sulfonylurinstoffer (SU) og Carvedilol.',
            'En 30-årig kvinde med multipel sklerose (MS) rapporterer følelsesløshed, muskelsvaghed og synsforstyrrelser. Hun tager i øjeblikket Interferon beta, Glatirameracetat og Methylprednisolon.',
            'Patienten er en 70-årig kvinde med osteoporose, oplever rygsmerter, en nedgang i højden og en nylig hoftefraktur. Hendes behandling inkluderer Alendronat, Calcium og Vitamin D-supplementer.',
            'En 52-årig mand diagnosticeret med hepatitis C præsenterer med træthed, gulsot og mavesmerter. Han behandles med Sofosbuvir, Ribavirin og Peginterferon alfa.',
            'Patienten er en 40-årig kvinde med migræne, rapporterer alvorlig hovedpine, kvalme og lysfølsomhed. Hun er ordineret Sumatriptan, Propranolol og Topiramat.',
            'En 65-årig mand med benign prostatahyperplasi (BPH) oplever vanskeligheder med at urinere, natlig vandladning og en svag urinstrøm. Hans medicin inkluderer Tamsulosin, Finasterid og Dutasterid.',
            'Patienten er en 28-årig kvinde med polycystisk ovariesyndrom (PCOS), lider af uregelmæssige perioder, hirsutisme og acne. Hun tager i øjeblikket Metformin, Spironolacton og orale præventionsmidler.',
            'En 58-årig mand diagnosticeret med kronisk hepatitis B præsenterer med træthed, mavesmerter og gulsot. Han behandles med Tenofovir, Entecavir og Lamivudin.',
            'Patienten er en 45-årig kvinde med hypothyroidisme, oplever vægtøgning, træthed og kuldefølsomhed. Hendes behandling inkluderer Levothyroxin, Liothyronin og Selen-supplementer.',
            'En 38-årig mand med cøliaki rapporterer diarré, vægttab og oppustethed. Hans medicin inkluderer en glutenfri kost, Loperamid og Vitamin D-supplementer.',
            'Patienten er en 50-årig kvinde med psoriasis, præsenterer med røde, skællede pletter på hendes hud, kløe og ledsmerter. Hun tager i øjeblikket Methotrexat, Adalimumab og Calcipotriol.',
            'En 60-årig mand diagnosticeret med urinsyregigt oplever alvorlige ledsmerter, hævelse og rødme, primært i hans storetå. Han behandles med Allopurinol, Colchicin og Indometacin.',
            'Patienten er en 27-årig kvinde med endometriose, lider af bækkensmerter, dysmenorré og infertilitet. Hendes medicin inkluderer Danazol, Leuprorelin og orale præventionsmidler.',
            'En 55-årig mand med kronisk hepatitis C præsenterer med træthed, mavesmerter og gulsot. Han behandles med Sofosbuvir, Ledipasvir og Ribavirin.',
            'Patienten er en 40-årig kvinde med Crohns sygdom, oplever mavesmerter, diarré og vægttab. Hendes behandling inkluderer Infliximab, Azathioprin og Mesalamin.',
            'En 68-årig mand diagnosticeret med prostatakræft rapporterer vanskeligheder med at urinere, hæmaturi og bækkenpine. Han tager i øjeblikket Leuprorelin, Flutamid og Bicalutamid.',
            'Patienten er en 33-årig kvinde med ankyloserende spondylitis, lider af kroniske rygsmerter, stivhed og nedsat mobilitet. Hendes medicin inkluderer Etanercept, Sulfasalazin og Naproxen.',
            'En 50-årig mand med amyotrofisk lateral sklerose (ALS) præsenterer med muskelsvaghed, talebesvær og åndedrætsproblemer. Hans behandling inkluderer Riluzol, Baclofen og Edaravon.'
        ]
    }

    # Creating the DataFrame
    df2 = pd.DataFrame(data2)

    df2.to_csv("dataset_notes/clin_note_danish_df.tsv", sep="\t", index=False)


if __name__ == "__main__":
    gen_dataframe()

