"""
Generate an English and a Danish DataFrames that contain patient id, codes 
and clinical notes.
"""
import pandas as pd

def gen_dataframe():
    """
    English data
    """
    # Define the clinical notes
    clinical_notes = [
    "The patient is a 55-year-old male diagnosed with hypertension. He presents with persistent headaches, episodes of dizziness, and blurred vision. His current medication regimen includes Lisinopril, Amlodipine, and Hydrochlorothiazide.",
    "A 62-year-old female with a history of type 2 diabetes reports polyuria, polydipsia, and ongoing fatigue. She is being treated with Metformin, Glipizide, and Insulin.",
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
    "The patient is a 55-year-old male diagnosed with type 2 diabetes, suffering from obesity, fatigue, and swelling in the legs. His current medication regimen includes Insulin, Sulfonylureas (SU), and Carvedilol.",
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
    "A 50-year-old male with amyotrophic lateral sclerosis (ALS) presents with muscle weakness, difficulty speaking, and respiratory problems. His treatment includes Riluzole, Baclofen, and Edaravone.",
    "A 58-year-old female diagnosed with hypertension. She presents with palpitations, shortness of breath, and chest pain. Her current medication regimen includes Losartan, Amlodipine, and Hydrochlorothiazide.",
    "A 65-year-old male with type 2 diabetes mellitus reports neuropathy, visual disturbances, and slow-healing wounds. He is being treated with Metformin, Insulin, and Pioglitazone.",
    "The patient is a 25-year-old female with asthma, experiencing nighttime cough, chest tightness, and shortness of breath. She is prescribed Salmeterol, Budesonide, and Montelukast.",
    "A 48-year-old male presents with rheumatoid arthritis, reporting swelling, decreased range of motion, and fatigue. His treatment includes Etanercept, Methotrexate, and Ibuprofen.",
    "The patient is a 70-year-old female with COPD, suffering from wheezing, chronic cough, and recurrent bronchitis. Her medications include Ipratropium, Salmeterol, and Prednisolone.",
    "A 35-year-old male with systemic lupus erythematosus presents with joint pain, fatigue, and a butterfly-shaped facial rash. He is currently taking Azathioprine, Hydroxychloroquine, and Prednisone.",
    "The patient is a 45-year-old female with GERD, experiencing regurgitation, sore throat, and a chronic cough. She is being treated with Esomeprazole, Famotidine, and Metoclopramide.",
    "A 29-year-old male diagnosed with IBS reports cramping, mucus in stool, and nausea. His medications include Hyoscyamine, Loperamide, and Amitriptyline.",
    "The patient is a 60-year-old female with Parkinson’s disease, presenting with balance issues, muscle stiffness, and speech difficulties. She is prescribed Ropinirole, Levodopa, and Amantadine.",
    "A 50-year-old male with CKD reports decreased appetite, nausea, and leg cramps. His treatment includes Losartan, Furosemide, and Erythropoietin.",
    "The patient is a 33-year-old female with ulcerative colitis, experiencing tenesmus, weight loss, and anemia. She is being treated with Balsalazide, Prednisone, and Azathioprine.",
    "A 47-year-old male with fibromyalgia presents with cognitive difficulties, widespread pain, and irritable bowel symptoms. His medications include Gabapentin, Duloxetine, and Tramadol."
    ]


    # Create a list of Patient IDs and codes
    patient_ids = [f"PID{1000+i}" for i in range(len(clinical_notes))]
    codes = [f"CODE{1000+i}" for i in range(len(clinical_notes))]

    # Create a dictionary
    data = {
        "patient_id": patient_ids,
        "code_id": codes,
        "clinical_notes": clinical_notes
    }
    # Convert into a DataFrame
    df = pd.DataFrame(data)
    # Save the English DataFrame
    df.to_csv("dataset_notes/clin_note_df.tsv", sep="\t", index=False)


    # Danish data
    clinical_notes_danish = [
    "Patienten er en 55-årig mand diagnosticeret med hypertension. Han præsenterer med vedvarende hovedpine, episoder af svimmelhed og sløret syn. Hans nuværende medicinregime inkluderer Lisinopril, Amlodipin og Hydrochlorothiazid.",
    "En 62-årig kvinde med en historie af type 2 diabetes rapporterer polyuri, polydipsi og vedvarende træthed. Hun behandles med Metformin, Glipizid og Insulin.",
    "Patienten er en 25-årig mand med astma, der oplever hvæsende vejrtrækning, åndenød og trykken for brystet. Han får ordineret Albuterol, Fluticason og Montelukast.",
    "En 48-årig kvinde præsenterer med leddegigt og rapporterer ledsmerter, morgenstivhed og hævelse. Hendes behandling inkluderer Methotrexat, Hydroxychloroquin og Prednison.",
    "Patienten er en 70-årig mand med kronisk obstruktiv lungesygdom (KOL), der lider af kronisk hoste, dyspnø og hyppige luftvejsinfektioner. Hans medicin inkluderer Tiotropium, Salbutamol og Prednisolon.",
    "En 35-årig kvinde med systemisk lupus erythematosus præsenterer med udslæt i ansigtet, ledsmerter og træthed. Hun tager i øjeblikket Hydroxychloroquin, Prednison og Mycophenolatmofetil.",
    "Patienten er en 45-årig mand med gastroøsofageal reflukssygdom (GERD), der oplever halsbrand, opstød og brystsmerter. Han behandles med Omeprazol, Ranitidin og Metoclopramid.",
    "En 29-årig kvinde diagnosticeret med irritabel tarmsyndrom (IBS) rapporterer mavesmerter, oppustethed og vekslende diarré og forstoppelse. Hendes medicin inkluderer Dicyclomin, Loperamid og Amitriptylin.",
    "Patienten er en 60-årig mand med Parkinsons sygdom, der præsenterer med rystelser, stivhed og bradykinesi. Han får ordineret Levodopa, Carbidopa og Pramipexol.",
    "En 50-årig kvinde med kronisk nyresygdom (CKD) rapporterer træthed, hævelse i benene og nedsat urinproduktion. Hendes behandling inkluderer Lisinopril, Furosemid og Erythropoietin.",
    "Patienten er en 33-årig mand med ulcerøs colitis, der oplever mavesmerter, blodig diarré og vægttab. Han behandles med Mesalamin, Prednison og Azathioprin.",
    "En 47-årig kvinde med fibromyalgi præsenterer med udbredte muskuloskeletale smerter, træthed og søvnforstyrrelser. Hendes medicin inkluderer Pregabalin, Duloxetin og Tramadol.",
    "Patienten er en 55-årig mand diagnosticeret med type 2 diabetes, der lider af fedme, træthed og hævelse i benene. Hans nuværende medicinregime inkluderer Insulin, Sulfonylurinstoffer (SU) og Carvedilol.",
    "En 30-årig kvinde med multipel sklerose (MS) rapporterer følelsesløshed, muskelsvaghed og synsforstyrrelser. Hun tager i øjeblikket Interferon beta, Glatirameracetat og Methylprednisolon.",
    "Patienten er en 70-årig kvinde med osteoporose, der oplever rygsmerter, nedsat højde og en nylig hoftefraktur. Hendes behandling inkluderer Alendronat, Calcium og Vitamin D kosttilskud.",
    "En 52-årig mand diagnosticeret med hepatitis C præsenterer med træthed, gulsot og mavesmerter. Han behandles med Sofosbuvir, Ribavirin og Peginterferon alfa.",
    "Patienten er en 40-årig kvinde med migræne, der rapporterer svær hovedpine, kvalme og lysfølsomhed. Hun får ordineret Sumatriptan, Propranolol og Topiramat.",
    "En 65-årig mand med benign prostatahyperplasi (BPH) oplever vandladningsbesvær, natlig vandladning og en svag urinstråle. Hans medicin inkluderer Tamsulosin, Finasterid og Dutasterid.",
    "Patienten er en 28-årig kvinde med polycystisk ovariesyndrom (PCOS), der lider af uregelmæssige menstruationer, hirsutisme og akne. Hun tager i øjeblikket Metformin, Spironolakton og orale præventionsmidler.",
    "En 58-årig mand diagnosticeret med kronisk hepatitis B præsenterer med træthed, mavesmerter og gulsot. Han behandles med Tenofovir, Entecavir og Lamivudin.",
    "Patienten er en 45-årig kvinde med hypothyroidisme, der oplever vægtøgning, træthed og kuldeintolerance. Hendes behandling inkluderer Levothyroxin, Liothyronin og Selen kosttilskud.",
    "En 38-årig mand med cøliaki rapporterer diarré, vægttab og oppustethed. Hans medicin inkluderer en glutenfri diæt, Loperamid og Vitamin D kosttilskud.",
    "Patienten er en 50-årig kvinde med psoriasis, der præsenterer med røde, skællende pletter på huden, kløe og ledsmerter. Hun tager i øjeblikket Methotrexat, Adalimumab og Calcipotriol.",
    "En 60-årig mand diagnosticeret med urinsyregigt oplever svære ledsmerter, hævelse og rødme, primært i storetåen. Han behandles med Allopurinol, Colchicin og Indometacin.",
    "Patienten er en 27-årig kvinde med endometriose, der lider af bækkensmerter, dysmenoré og infertilitet. Hendes medicin inkluderer Danazol, Leuprolid og orale præventionsmidler.",
    "En 55-årig mand med kronisk hepatitis C præsenterer med træthed, mavesmerter og gulsot. Han behandles med Sofosbuvir, Ledipasvir og Ribavirin.",
    "Patienten er en 40-årig kvinde med Crohns sygdom, der oplever mavesmerter, diarré og vægttab. Hendes behandling inkluderer Infliximab, Azathioprin og Mesalamin.",
    "En 68-årig mand diagnosticeret med prostatakræft rapporterer vandladningsbesvær, hæmaturi og bækkensmerter. Han tager i øjeblikket Leuprolid, Flutamid og Bikalutamid.",
    "Patienten er en 33-årig kvinde med ankyloserende spondylitis, der lider af kroniske rygsmerter, stivhed og nedsat bevægelighed. Hendes medicin inkluderer Etanercept, Sulfasalazin og Naproxen.",
    "En 50-årig mand med amyotrofisk lateral sklerose (ALS) præsenterer med muskelsvaghed, talebesvær og vejrtrækningsproblemer. Hans behandling inkluderer Riluzol, Baclofen og Edaravone.",
    "En 58-årig kvinde diagnosticeret med hypertension. Hun præsenterer med hjertebanken, åndenød og brystsmerter. Hendes nuværende medicinregime inkluderer Losartan, Amlodipin og Hydrochlorothiazid.",
    "En 65-årig mand med type 2 diabetes rapporterer neuropati, synsforstyrrelser og langsomt helende sår. Han behandles med Metformin, Insulin og Pioglitazon.",
    "Patienten er en 25-årig kvinde med astma, der oplever natlig hoste, trykken for brystet og åndenød. Hun får ordineret Salmeterol, Budesonid og Montelukast.",
    "En 48-årig mand præsenterer med leddegigt og rapporterer hævelse, nedsat bevægelighed og træthed. Hans behandling inkluderer Etanercept, Methotrexat og Ibuprofen.",
    "Patienten er en 70-årig kvinde med KOL, der lider af hvæsende vejrtrækning, kronisk hoste og tilbagevendende bronkitis. Hendes medicin inkluderer Ipratropium, Salmeterol og Prednisolon.",
    "En 35-årig mand med systemisk lupus erythematosus præsenterer med ledsmerter, træthed og et sommerfugleformet udslæt i ansigtet. Han tager i øjeblikket Azathioprin, Hydroxychloroquin og Prednison.",
    "Patienten er en 45-årig kvinde med GERD, der oplever opstød, ondt i halsen og kronisk hoste. Hun behandles med Esomeprazol, Famotidin og Metoclopramid.",
    "En 29-årig mand diagnosticeret med IBS rapporterer kramper, slim i afføringen og kvalme. Hans medicin inkluderer Hyoscyamin, Loperamid og Amitriptylin.",
    "Patienten er en 60-årig kvinde med Parkinsons sygdom, der præsenterer med balanceproblemer, muskelstivhed og talevanskeligheder. Hun får ordineret Ropinirol, Levodopa og Amantadin.",
    "En 50-årig mand med CKD rapporterer nedsat appetit, kvalme og kramper i benene. Hans behandling inkluderer Losartan, Furosemid og Erythropoietin.",
    "Patienten er en 33-årig kvinde med ulcerøs colitis, der oplever tenesmus, vægttab og anæmi. Hun behandles med Balsalazid, Prednison og Azathioprin.",
    "En 47-årig mand med fibromyalgi præsenterer med kognitive vanskeligheder, udbredte smerter og irritabel tarmsymptomer. Hans medicin inkluderer Gabapentin, Duloxetin og Tramadol."
    ]


    # Create a list of Patient IDs and codes
    patient_ids = [f"PID{1000+i}" for i in range(len(clinical_notes))]
    codes = [f"CODE{1000+i}" for i in range(len(clinical_notes))]

    # Create a dictionary
    data2 = {
        "patient_id": patient_ids,
        "code_id": codes,
        "clinical_notes": clinical_notes_danish
    }
    
    # Creating the DataFrame
    df2 = pd.DataFrame(data2)
    # Save the Danish DataFrame
    df2.to_csv("dataset_notes/clin_note_danish_df.tsv", sep="\t", index=False)


if __name__ == "__main__":
    gen_dataframe()